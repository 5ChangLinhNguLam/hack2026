"""End-to-end causal inference over one cabin frame at a time."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn

from .data.eye_rois import crop_eye_pair
from .data.eye_sequences import PHASE_NAMES
from .data.primitive_dataset import letterbox, normalized_image_tensor
from .distraction import (
    action_distraction_probability,
    pool_distraction_probabilities,
)
from .landmarks import (
    MOUTH_INDICES,
    FaceLandmarkDetector,
    eye_roi_record_from_landmarks,
    mouth_crop_from_landmarks,
)
from .runtime_fusion import PrimitiveEvidence, RuntimeEvidenceFusion, RuntimeResult
from .state.arbiter import DriverState
from .state.perclos import OnlinePerclos


_POSE_INDICES = (1, 152, 33, 263, 61, 291)
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class FramePrediction:
    result: RuntimeResult
    eye_phase: str
    closed_probability: float
    yawn_probability: float
    distraction_probability: float
    head_pitch: float
    head_yaw: float
    state_probabilities: tuple[float, ...] = ()


def _landmark_quality(landmarks: Sequence[object] | None, indices: Sequence[int]) -> float:
    if landmarks is None:
        return 0.0
    values: list[float] = []
    for index in indices:
        landmark = landmarks[index]
        visibility = getattr(landmark, "visibility", None)
        presence = getattr(landmark, "presence", None)
        candidates = [value for value in (visibility, presence) if value is not None]
        values.append(min(float(value) for value in candidates) if candidates else 1.0)
    return max(0.0, min(1.0, sum(values) / len(values)))


def _face_crop(
    image: Image.Image,
    landmarks: Sequence[object] | None,
    *,
    size: int = 224,
    padding: float = 0.35,
) -> tuple[Image.Image, float]:
    if landmarks is None:
        return Image.new("RGB", (size, size)), 0.0
    xs = [float(getattr(landmark, "x")) * image.width for landmark in landmarks]
    ys = [float(getattr(landmark, "y")) * image.height for landmark in landmarks]
    width, height = max(xs) - min(xs), max(ys) - min(ys)
    margin = max(width, height) * padding
    x1, y1 = max(0, round(min(xs) - margin)), max(0, round(min(ys) - margin))
    x2 = min(image.width, round(max(xs) + margin))
    y2 = min(image.height, round(max(ys) + margin))
    if x2 <= x1 or y2 <= y1:
        return Image.new("RGB", (size, size)), 0.0
    crop = image.crop((x1, y1, x2, y2)).resize((size, size), Image.Resampling.BILINEAR)
    return crop, _landmark_quality(landmarks, range(len(landmarks)))


def estimate_head_pose(
    landmarks: Sequence[object] | None, image_width: int, image_height: int
) -> tuple[float, float]:
    if landmarks is None or len(landmarks) <= max(_POSE_INDICES):
        return 0.0, 0.0
    try:
        import cv2
    except ImportError:  # pragma: no cover - deployment has OpenCV
        return 0.0, 0.0
    image_points = np.array(
        [
            (
                float(getattr(landmarks[index], "x")) * image_width,
                float(getattr(landmarks[index], "y")) * image_height,
            )
            for index in _POSE_INDICES
        ],
        dtype=np.float64,
    )
    focal = float(image_width)
    camera = np.array(
        [[focal, 0, image_width / 2], [0, focal, image_height / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    ok, rotation_vector, _ = cv2.solvePnP(
        _MODEL_POINTS,
        image_points,
        camera,
        np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0
    rotation, _ = cv2.Rodrigues(rotation_vector)
    sy = float(np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2))
    pitch = float(np.degrees(np.arctan2(-rotation[2, 0], sy)))
    yaw = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    return pitch, yaw


class NodTracker:
    def __init__(self, fps: float, window_seconds: float = 1.5) -> None:
        self.values: deque[float] = deque(maxlen=max(3, round(fps * window_seconds)))

    def update(self, pitch: float, visible: bool) -> float:
        if not visible:
            self.values.clear()
            return 0.0
        self.values.append(pitch)
        if len(self.values) < 6:
            return 0.0
        values = tuple(self.values)
        amplitude = max(values) - min(values)
        returned = abs(values[-1] - values[0]) <= max(3.0, 0.35 * amplitude)
        return min(amplitude / (18.0 if returned else 36.0), 1.0)


def _eye_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div_(127.5).sub_(1.0)


def _calibrated_evidence(
    logit: float,
    calibration: Mapping[str, float] | None,
    *,
    decision_level: float = 0.7,
) -> float:
    """Map a selected validation threshold onto the arbiter decision level."""
    values = calibration or {}
    temperature = max(float(values.get("temperature", 1.0)), 1e-6)
    threshold = min(max(float(values.get("threshold", 0.5)), 1e-6), 1.0 - 1e-6)
    target = min(max(decision_level, 1e-6), 1.0 - 1e-6)
    shifted = (
        logit / temperature
        - math.log(threshold / (1.0 - threshold))
        + math.log(target / (1.0 - target))
    )
    if shifted >= 0.0:
        factor = math.exp(-shifted)
        return 1.0 / (1.0 + factor)
    factor = math.exp(shifted)
    return factor / (1.0 + factor)


class RealtimeDriverPipeline:
    def __init__(
        self,
        *,
        eye_model: nn.Module,
        primitive_model: nn.Module,
        mouth_model: nn.Module | None = None,
        cabin_temporal_model: nn.Module | None = None,
        face_mouth_temporal_model: nn.Module | None = None,
        unified_model: nn.Module | None = None,
        cabin_calibration: Mapping[str, Mapping[str, float]] | None = None,
        face_mouth_calibration: Mapping[str, Mapping[str, float]] | None = None,
        landmark_detector: FaceLandmarkDetector,
        device: torch.device,
        fps: float = 20.0,
        sequence_length: int = 15,
        confirmation_frames: int = 3,
        primitive_interval_frames: int = 1,
        temporal_sequence_length: int = 20,
        unified_sequence_length: int = 100,
    ) -> None:
        if primitive_interval_frames <= 0:
            raise ValueError("primitive_interval_frames must be positive")
        specialist_models = (cabin_temporal_model, face_mouth_temporal_model)
        if any(model is not None for model in specialist_models) and not (
            mouth_model is not None and all(model is not None for model in specialist_models)
        ):
            raise ValueError("mouth, cabin temporal, and face-mouth temporal models must be supplied together")
        if unified_model is not None and mouth_model is None:
            raise ValueError("the unified five-state model requires the mouth encoder")
        if temporal_sequence_length <= 0:
            raise ValueError("temporal_sequence_length must be positive")
        if unified_sequence_length <= 0:
            raise ValueError("unified_sequence_length must be positive")
        self.eye_model = eye_model.to(device).eval()
        self.primitive_model = primitive_model.to(device).eval()
        self.mouth_model = mouth_model.to(device).eval() if mouth_model is not None else None
        self.cabin_temporal_model = (
            cabin_temporal_model.to(device).eval()
            if cabin_temporal_model is not None
            else None
        )
        self.face_mouth_temporal_model = (
            face_mouth_temporal_model.to(device).eval()
            if face_mouth_temporal_model is not None
            else None
        )
        self.unified_model = (
            unified_model.to(device).eval() if unified_model is not None else None
        )
        self.temporal_enabled = (
            mouth_model is not None
            and cabin_temporal_model is not None
            and face_mouth_temporal_model is not None
        )
        self.unified_enabled = self.unified_model is not None
        self.cabin_calibration = dict(cabin_calibration or {})
        self.face_mouth_calibration = dict(face_mouth_calibration or {})
        self.landmark_detector = landmark_detector
        self.device = device
        self.eye_history: deque[tuple[Tensor, Tensor]] = deque(maxlen=sequence_length)
        self.fusion = RuntimeEvidenceFusion(
            fps=fps, confirmation_frames=confirmation_frames
        )
        self.nod = NodTracker(fps)
        self.primitive_interval_frames = primitive_interval_frames
        self._frame_index = 0
        self._primitive_cache: tuple[float, float] | None = None
        self.temporal_sequence_length = temporal_sequence_length
        self.cabin_embedding_history: deque[Tensor] = deque(
            maxlen=temporal_sequence_length
        )
        self.face_mouth_embedding_history: deque[Tensor] = deque(
            maxlen=temporal_sequence_length
        )
        self.unified_sequence_length = unified_sequence_length
        self.unified_embedding_history: deque[Tensor] = deque(
            maxlen=unified_sequence_length
        )
        self.unified_perclos = OnlinePerclos(
            fps=fps, windows_seconds=(10.0, 30.0, 60.0)
        )
        self._unified_cache: tuple[RuntimeResult, tuple[float, ...]] | None = None

    def reset(self) -> None:
        self.eye_history.clear()
        self.fusion.reset()
        self.nod.values.clear()
        self._frame_index = 0
        self._primitive_cache = None
        self.cabin_embedding_history.clear()
        self.face_mouth_embedding_history.clear()
        self.unified_embedding_history.clear()
        self.unified_perclos.reset()
        self._unified_cache = None

    def _history_window(
        self, history: deque[Tensor], *, length: int | None = None
    ) -> Tensor:
        values = torch.stack(tuple(history), dim=0)
        expected = self.temporal_sequence_length if length is None else length
        pad = expected - len(values)
        if pad:
            values = torch.cat(
                (
                    torch.zeros(
                        pad,
                        values.shape[-1],
                        dtype=values.dtype,
                        device=values.device,
                    ),
                    values,
                ),
                dim=0,
            )
        return values.unsqueeze(0)

    def close(self) -> None:
        close = getattr(self.landmark_detector, "close", None)
        if close is not None:
            close()

    @torch.inference_mode()
    def process(self, image: Image.Image, *, timestamp: float | None = None) -> FramePrediction:
        image = image.convert("RGB")
        landmarks = self.landmark_detector.detect(image)
        eye_record = eye_roi_record_from_landmarks(
            frame_id=0,
            landmarks=landmarks,
            image_width=image.width,
            image_height=image.height,
        )
        left, right, eye_visibility_values = crop_eye_pair(
            image, eye_record, output_size=(96, 48)
        )
        eye_pair = torch.stack((_eye_tensor(left), _eye_tensor(right)))
        eye_visibility = torch.tensor(eye_visibility_values, dtype=torch.float32)
        self.eye_history.append((eye_pair, eye_visibility))
        eyes = torch.stack([item[0] for item in self.eye_history]).unsqueeze(0).to(self.device)
        visibility = torch.stack([item[1] for item in self.eye_history]).unsqueeze(0).to(self.device)

        pitch, yaw = estimate_head_pose(landmarks, image.width, image.height)
        face_visibility = _landmark_quality(landmarks, range(len(landmarks))) if landmarks else 0.0
        active_cache = self._unified_cache if self.unified_enabled else self._primitive_cache
        run_primitives = (
            active_cache is None
            or self._frame_index % self.primitive_interval_frames == 0
        )
        if run_primitives:
            face_crop, face_visibility = _face_crop(image, landmarks)
            mouth_crop, mouth_visibility = mouth_crop_from_landmarks(image, landmarks)
            cabin_image = letterbox(image, (320, 192))
            cabin = normalized_image_tensor(cabin_image, (320, 192)).unsqueeze(0).to(self.device)
            face = normalized_image_tensor(face_crop, (224, 224)).unsqueeze(0).to(self.device)
            face_visibility_tensor = torch.tensor([face_visibility], device=self.device)
            mouth_visibility_tensor = torch.tensor(
                [mouth_visibility], dtype=torch.float32, device=self.device
            )
            pose = torch.tensor([[pitch, yaw]], dtype=torch.float32, device=self.device)
            if self.temporal_enabled or self.unified_enabled:
                mouth = _eye_tensor(mouth_crop).unsqueeze(0).to(self.device)

        use_amp = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, enabled=use_amp):
            if self.unified_enabled:
                eye_visual = self.eye_model.encode_visual(eyes, visibility)
                eye_temporal_features = self.eye_model.encode_temporal(eye_visual)
                phase_logits = self.eye_model.phase_head(eye_temporal_features)
                closedness_logits = self.eye_model.closedness_head(
                    eye_temporal_features
                ).squeeze(-1)
            else:
                eye_output = self.eye_model(eyes, visibility)
                phase_logits = eye_output.phase_logits
                closedness_logits = eye_output.closedness_logits
        phase_probabilities = phase_logits[0, -1].float().softmax(dim=0)
        phase_index = int(phase_probabilities.argmax())
        closed_probability = float(closedness_logits[0, -1].float().sigmoid())
        eye_quality = max(eye_visibility_values)
        perclos_snapshot = (
            self.unified_perclos.update(
                closed_probability=closed_probability,
                visibility=eye_quality,
                timestamp=timestamp,
            )
            if self.unified_enabled
            else None
        )

        with torch.autocast(device_type=self.device.type, enabled=use_amp):
            if run_primitives:
                if self.unified_enabled:
                    assert self.mouth_model is not None
                    assert self.unified_model is not None
                    assert perclos_snapshot is not None
                    cabin_embedding = self.primitive_model.encode_cabin(cabin)
                    face_embedding = self.primitive_model.encode_face(
                        face, face_visibility_tensor, pose
                    )
                    mouth_embedding = self.mouth_model.encode(mouth)
                    mouth_embedding = mouth_embedding * mouth_visibility_tensor.reshape(-1, 1)
                    normalized_pose = pose.clamp(-90.0, 90.0) / 90.0
                    windows = (10.0, 30.0, 60.0)
                    eye_evidence = torch.tensor(
                        [
                            closed_probability,
                            perclos_snapshot.closure_duration_seconds,
                            *(
                                0.0
                                if perclos_snapshot.slow_values[window] is None
                                else float(perclos_snapshot.slow_values[window])
                                for window in windows
                            ),
                            *(
                                float(perclos_snapshot.reliable_by_window[window])
                                for window in windows
                            ),
                        ],
                        dtype=face_embedding.dtype,
                        device=self.device,
                    ).reshape(1, 8)
                    fused_embedding = torch.cat(
                        (
                            cabin_embedding,
                            face_embedding,
                            mouth_embedding,
                            eye_temporal_features[:, -1],
                            face_visibility_tensor.reshape(-1, 1).to(face_embedding.dtype),
                            mouth_visibility_tensor.reshape(-1, 1).to(face_embedding.dtype),
                            visibility[:, -1].to(face_embedding.dtype),
                            normalized_pose.to(face_embedding.dtype),
                            eye_evidence,
                        ),
                        dim=-1,
                    )
                    self.unified_embedding_history.append(fused_embedding[0].float())
                    unified_output = self.unified_model(
                        self._history_window(
                            self.unified_embedding_history,
                            length=self.unified_sequence_length,
                        )
                    )
                elif self.temporal_enabled:
                    assert self.mouth_model is not None
                    assert self.cabin_temporal_model is not None
                    assert self.face_mouth_temporal_model is not None
                    cabin_embedding = self.primitive_model.encode_cabin(cabin)
                    face_embedding = self.primitive_model.encode_face(
                        face, face_visibility_tensor, pose
                    )
                    mouth_embedding = self.mouth_model.encode(mouth)
                    mouth_embedding = mouth_embedding * mouth_visibility_tensor.reshape(-1, 1)
                    normalized_pose = pose.clamp(-90.0, 90.0) / 90.0
                    face_mouth_embedding = torch.cat(
                        (
                            face_embedding,
                            mouth_embedding,
                            face_visibility_tensor.reshape(-1, 1).to(face_embedding.dtype),
                            mouth_visibility_tensor.reshape(-1, 1).to(face_embedding.dtype),
                            normalized_pose.to(face_embedding.dtype),
                        ),
                        dim=-1,
                    )
                    self.cabin_embedding_history.append(cabin_embedding[0].float())
                    self.face_mouth_embedding_history.append(
                        face_mouth_embedding[0].float()
                    )
                    cabin_temporal_output = self.cabin_temporal_model(
                        self._history_window(self.cabin_embedding_history)
                    )
                    face_mouth_temporal_output = self.face_mouth_temporal_model(
                        self._history_window(self.face_mouth_embedding_history)
                    )
                else:
                    primitive_output = self.primitive_model(
                        cabin, face, face_visibility_tensor, pose
                    )

        if self.unified_enabled:
            if run_primitives:
                state_probabilities = tuple(
                    float(value)
                    for value in unified_output.logits[0, -1].float().softmax(dim=-1)
                )
                state_id = max(
                    range(len(state_probabilities)),
                    key=state_probabilities.__getitem__,
                )
                states = (
                    DriverState.ALERT,
                    DriverState.DROWSY,
                    DriverState.MICROSLEEP,
                    DriverState.YAWNING,
                    DriverState.DISTRACTION,
                )
                assert perclos_snapshot is not None
                result = RuntimeResult(
                    state_id=state_id,
                    state=states[state_id],
                    confidence=state_probabilities[state_id],
                    visible=eye_quality >= 0.2 or face_visibility >= 0.2,
                    diagnostics={
                        "closed_probability": closed_probability,
                        "closure_duration_seconds": perclos_snapshot.closure_duration_seconds,
                        "perclos_10": float(perclos_snapshot.slow_values[10.0] or 0.0),
                        "perclos_30": float(perclos_snapshot.slow_values[30.0] or 0.0),
                        "perclos_60": float(perclos_snapshot.slow_values[60.0] or 0.0),
                    },
                )
                self._unified_cache = (result, state_probabilities)
            else:
                assert self._unified_cache is not None
                result, state_probabilities = self._unified_cache
            self._frame_index += 1
            return FramePrediction(
                result=result,
                eye_phase=PHASE_NAMES[phase_index],
                closed_probability=closed_probability,
                yawn_probability=state_probabilities[3],
                distraction_probability=state_probabilities[4],
                head_pitch=pitch,
                head_yaw=yaw,
                state_probabilities=state_probabilities,
            )

        if run_primitives:
            if self.temporal_enabled:
                direct_yawn = _calibrated_evidence(
                    float(face_mouth_temporal_output.binary_yawn_logits[0, -1].float()),
                    self.face_mouth_calibration.get("binary_yawn"),
                )
                type_yawn_probability = face_mouth_temporal_output.yawn_type_logits[
                    0, -1
                ].float().softmax(dim=-1)[1:].sum()
                type_yawn_logit = float(
                    torch.logit(type_yawn_probability.clamp(1e-6, 1.0 - 1e-6))
                )
                derived_yawn = _calibrated_evidence(
                    type_yawn_logit,
                    self.face_mouth_calibration.get("type_derived_yawn"),
                )
                yawn_probability = (2.0 * direct_yawn + derived_yawn) / 3.0

                direct_distraction = _calibrated_evidence(
                    float(cabin_temporal_output.distraction_logits[0, -1].float()),
                    self.cabin_calibration.get("direct_distraction"),
                )
                action_probability = action_distraction_probability(
                    cabin_temporal_output.action_logits[0, -1].float()
                )
                action_logit = float(
                    torch.logit(action_probability.clamp(1e-6, 1.0 - 1e-6))
                )
                action_distraction = _calibrated_evidence(
                    action_logit,
                    self.cabin_calibration.get("action_distraction"),
                )
                off_road = _calibrated_evidence(
                    float(cabin_temporal_output.road_gaze_logits[0, -1].float()),
                    self.cabin_calibration.get("road_gaze"),
                )
                pooled = float(
                    pool_distraction_probabilities(
                        direct_distraction, action_distraction, off_road
                    )
                )
                pooled_logit = math.log(
                    max(pooled, 1e-6) / max(1.0 - pooled, 1e-6)
                )
                distraction_probability = _calibrated_evidence(
                    pooled_logit,
                    self.cabin_calibration.get("fused_distraction"),
                )
            else:
                yawn_probability = float(
                    primitive_output["yawn"][0].float().softmax(dim=0)[1:].sum()
                )
                direct_distraction = float(
                    primitive_output["distraction"][0].float().softmax(dim=0)[1]
                )
                off_road = float(
                    primitive_output["road_gaze"][0].float().softmax(dim=0)[1]
                )
                action_distraction = float(
                    action_distraction_probability(
                        primitive_output["driver_action"][0].float()
                    )
                )
                distraction_probability = float(
                    pool_distraction_probabilities(
                        direct_distraction, action_distraction, off_road
                    )
                )
            self._primitive_cache = (yawn_probability, distraction_probability)
        else:
            assert self._primitive_cache is not None
            yawn_probability, distraction_probability = self._primitive_cache
        self._frame_index += 1
        mouth_visibility = _landmark_quality(landmarks, MOUTH_INDICES)
        nod_probability = self.nod.update(pitch, face_visibility >= 0.2)
        result = self.fusion.update(
            PrimitiveEvidence(
                closed_probability=closed_probability,
                eye_visibility=eye_quality,
                yawn_probability=yawn_probability,
                mouth_visibility=mouth_visibility,
                distraction_probability=distraction_probability,
                nod_probability=nod_probability,
                timestamp=timestamp,
            )
        )
        return FramePrediction(
            result=result,
            eye_phase=PHASE_NAMES[phase_index],
            closed_probability=closed_probability,
            yawn_probability=yawn_probability,
            distraction_probability=distraction_probability,
            head_pitch=pitch,
            head_yaw=yaw,
        )
