from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

from PIL import Image
import torch

from ..config import EyeTemporalConfig
from ..data.eye_sequences import PHASE_NAMES
from ..data.five_state_labels import FiveState
from ..face_detection import (
    PeriodicFaceDetector,
    TrackedFaceDetector,
    UltraLightFaceDetector,
)
from ..landmarks import (
    AsyncFaceLandmarkDetector,
    FaceBoxLandmarkDetector,
    MediaPipeFaceLandmarker,
    evidence_regions_from_landmarks,
)
from ..models.mobilenet_lstm import (
    CausalFourStateLSTM,
    MobileNetV3LargeVisualEncoder,
)
from ..models.ocular_lstm import CausalOcularLSTM
from ..models.eye_temporal import EyeTemporalNet
from ..models.embedding_temporal import CabinEmbeddingTCN, FaceMouthEmbeddingTCN
from ..models.five_state_temporal import FiveStateFusionTCN
from ..models.mouth_visual import MouthVisualNet
from ..models.spatial_multitask import SpatialPrimitiveNet
from ..runtime import RealtimeDriverPipeline
from ..runtime_mobilenet_lstm import (
    MobileNetLSTMRuntime,
    RuntimeFiveStatePrediction,
)
from ..state.causal_evidence import EVIDENCE_FEATURE_NAMES
from ..training.ocular_trainer import OCULAR_CHECKPOINT_SCHEMA_VERSION
from .precompute_embeddings import checkpoint_fingerprint


def _normalized_split(payload: dict[str, object]) -> dict[str, object]:
    raw = payload.get("split")
    if not isinstance(raw, dict):
        raise ValueError("checkpoint is missing split metadata")
    return {
        "train_subjects": tuple(str(value) for value in raw["train_subjects"]),
        "validation_subject": str(raw["validation_subject"]),
        "test_subject": str(raw["test_subject"]),
    }


def mouth_split_compatible(
    mouth_split: dict[str, object], spatial_split: dict[str, object]
) -> bool:
    """Mouth training omits fold subjects that have no s5 protocol."""
    return (
        mouth_split["validation_subject"] == spatial_split["validation_subject"]
        and mouth_split["test_subject"] == spatial_split["test_subject"]
        and bool(mouth_split["train_subjects"])
        and set(mouth_split["train_subjects"]).issubset(
            set(spatial_split["train_subjects"])
        )
    )


@dataclass
class LSTMDemoPipeline:
    runtime: MobileNetLSTMRuntime
    region_detector: FaceBoxLandmarkDetector

    def process(
        self,
        image: Image.Image,
        *,
        timestamp: float | None = None,
    ) -> RuntimeFiveStatePrediction:
        rgb = image.convert("RGB")
        landmarks = self.region_detector.detect(rgb)
        detection = self.region_detector.latest_detection
        regions = evidence_regions_from_landmarks(
            frame_id=0,
            face_box=None if detection is None else detection.box,
            landmarks=landmarks,
            image_width=rgb.width,
            image_height=rgb.height,
            pitch=0.0,
            yaw=0.0,
            face_visibility=(
                0.0 if detection is None else detection.confidence
            ),
        )
        return self.runtime.process(
            rgb,
            region_boxes=regions.boxes,
            region_visibility=regions.visibility,
            timestamp=timestamp,
        )

    def close(self) -> None:
        self.region_detector.close()


def _runtime_flags_for_temporal_schema(
    schema: int,
) -> tuple[bool, bool, bool]:
    """Return phase, event-gate, and fatigue-episode runtime flags."""

    return (
        schema in (5, 6, 7),
        schema in (6, 7),
        schema == 7,
    )


def create_lstm_pipeline(
    *,
    visual_checkpoint: Path,
    ocular_checkpoint: Path,
    temporal_checkpoint: Path,
    ultralight_model: Path,
    landmark_model: Path,
    device: torch.device,
    fps: float,
    detector_interval: int,
    gate_ocular_checkpoint: Path | None = None,
) -> LSTMDemoPipeline:
    visual_payload = torch.load(
        visual_checkpoint,
        map_location=device,
        weights_only=False,
    )
    temporal_payload = torch.load(
        temporal_checkpoint,
        map_location=device,
        weights_only=False,
    )
    ocular_payload = torch.load(
        ocular_checkpoint,
        map_location=device,
        weights_only=False,
    )
    gate_ocular_payload = (
        None
        if gate_ocular_checkpoint is None
        else torch.load(
            gate_ocular_checkpoint,
            map_location=device,
            weights_only=False,
        )
    )
    if visual_payload.get("schema_version") != 3:
        raise ValueError("visual checkpoint must use evidence schema 3")
    if (
        ocular_payload.get("schema_version")
        != OCULAR_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError(
            "ocular checkpoint must use ocular schema "
            f"{OCULAR_CHECKPOINT_SCHEMA_VERSION}"
        )
    temporal_schema = int(temporal_payload.get("schema_version", 0))
    if (
        temporal_schema not in (3, 4, 5, 6, 7)
        or temporal_payload.get("model_type") != "causal_four_state_lstm"
    ):
        raise ValueError(
            "temporal checkpoint must be a schema-3 four-state, schema-4 "
            "dual-ocular, schema-5 phase-aware, schema-6 causal-drowsy, "
            "or schema-7 fatigue-episode LSTM"
        )
    phase_aware, causal_drowsy, fatigue_episode = (
        _runtime_flags_for_temporal_schema(temporal_schema)
    )
    dual_ocular = (
        gate_ocular_payload is not None
        if phase_aware
        else temporal_schema == 4
    )
    if (temporal_schema == 4) != (gate_ocular_payload is not None) and not phase_aware:
        raise ValueError(
            "schema-4 temporal checkpoints require a separate microsleep-gate "
            "ocular checkpoint; schema-3 checkpoints do not accept one"
        )
    if gate_ocular_payload is not None and (
        gate_ocular_payload.get("schema_version")
        != OCULAR_CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError(
            "microsleep-gate ocular checkpoint must use ocular schema "
            f"{OCULAR_CHECKPOINT_SCHEMA_VERSION}"
        )
    split_payloads = [visual_payload, ocular_payload, temporal_payload]
    if gate_ocular_payload is not None:
        split_payloads.append(gate_ocular_payload)
    splits = tuple(
        _normalized_split(payload) for payload in split_payloads
    )
    if len({json.dumps(value, sort_keys=True) for value in splits}) != 1:
        raise ValueError("visual, ocular, and LSTM checkpoint split mismatch")
    visual_fingerprint = checkpoint_fingerprint(visual_checkpoint)
    ocular_fingerprint = checkpoint_fingerprint(ocular_checkpoint)
    gate_ocular_fingerprint = (
        None
        if gate_ocular_checkpoint is None
        else checkpoint_fingerprint(gate_ocular_checkpoint)
    )
    if ocular_payload.get("visual_checkpoint_fingerprint") != visual_fingerprint:
        raise ValueError("ocular checkpoint visual fingerprint mismatch")
    if temporal_payload.get("visual_checkpoint_fingerprint") != visual_fingerprint:
        raise ValueError("LSTM checkpoint visual fingerprint mismatch")
    if temporal_payload.get("ocular_checkpoint_fingerprint") != ocular_fingerprint:
        raise ValueError("LSTM checkpoint ocular fingerprint mismatch")
    if temporal_payload.get(
        "microsleep_gate_ocular_checkpoint_fingerprint"
    ) != gate_ocular_fingerprint:
        raise ValueError(
            "LSTM checkpoint microsleep-gate ocular fingerprint mismatch"
        )
    if gate_ocular_payload is not None and (
        gate_ocular_payload.get("visual_checkpoint_fingerprint")
        != visual_fingerprint
    ):
        raise ValueError(
            "microsleep-gate ocular checkpoint visual fingerprint mismatch"
        )
    if temporal_payload.get("evidence_feature_names") != list(
        EVIDENCE_FEATURE_NAMES
    ):
        raise ValueError("LSTM checkpoint evidence feature contract mismatch")
    if phase_aware and temporal_payload.get(
        "ocular_phase_feature_names"
    ) != list(PHASE_NAMES):
        raise ValueError("LSTM checkpoint ocular phase contract mismatch")
    if fatigue_episode:
        expected_fusion = (
            "dual_ocular_fatigue_episode_and_microsleep_gates"
            if dual_ocular
            else "fatigue_episode_and_microsleep_gates"
        )
    elif causal_drowsy:
        expected_fusion = (
            "dual_ocular_drowsy_and_microsleep_gates"
            if dual_ocular
            else "drowsy_and_microsleep_gates"
        )
    else:
        expected_fusion = (
            "dual_ocular_deterministic_two_second_gate"
            if dual_ocular
            else "deterministic_two_second_gate"
        )
    if temporal_payload.get("microsleep_fusion") != expected_fusion:
        raise ValueError("LSTM checkpoint microsleep fusion contract mismatch")
    evidence_config = temporal_payload.get("evidence_config")
    if not isinstance(evidence_config, dict) or not math.isclose(
        float(evidence_config.get("microsleep_seconds", 0.0)),
        2.0,
    ):
        raise ValueError("LSTM checkpoint must use the two-second event boundary")
    fatigue_episode_seconds = float(
        evidence_config.get("fatigue_episode_seconds", 0.0)
    )
    fatigue_yawn_block_threshold = float(
        evidence_config.get("fatigue_episode_yawn_block_threshold", 0.7)
    )
    fatigue_distraction_block_threshold = float(
        evidence_config.get(
            "fatigue_episode_distraction_block_threshold",
            0.7,
        )
    )
    if fatigue_episode and (
        not math.isfinite(fatigue_episode_seconds)
        or fatigue_episode_seconds <= 0.0
    ):
        raise ValueError(
            "schema-7 LSTM checkpoint requires a positive fatigue episode duration"
        )
    if fatigue_episode and any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in (
            fatigue_yawn_block_threshold,
            fatigue_distraction_block_threshold,
        )
    ):
        raise ValueError(
            "schema-7 fatigue episode conflict thresholds must be in [0, 1]"
        )
    target_fps = tuple(
        float(payload.get("target_fps", 0.0)) for payload in split_payloads
    )
    if any(value <= 0.0 for value in target_fps) or not all(
        math.isclose(value, target_fps[0]) for value in target_fps[1:]
    ):
        raise ValueError("visual, ocular, and LSTM checkpoint FPS mismatch")
    if not math.isclose(float(fps), target_fps[0]):
        raise ValueError("demo FPS must match the checkpoint target FPS")

    visual_config = visual_payload["model_config"]
    if visual_config.get("region_order") != [
        "face",
        "left_eye",
        "right_eye",
        "mouth",
    ]:
        raise ValueError("visual checkpoint region order mismatch")
    embedding_dim = int(visual_config["embedding_dim"])
    region_dim = int(visual_config["region_dim"])
    visual_embedding_dim = embedding_dim + 4 * region_dim
    encoder = MobileNetV3LargeVisualEncoder(
        embedding_dim=embedding_dim,
        region_dim=region_dim,
        pretrained=False,
    )
    encoder.load_state_dict(visual_payload["model_state"])
    ocular_config = ocular_payload["model_config"]
    if int(ocular_config.get("input_dim", 0)) != 2 * region_dim + 5:
        raise ValueError("ocular checkpoint input dimension mismatch")
    ocular = CausalOcularLSTM(
        input_dim=int(ocular_config["input_dim"]),
        projection_dim=int(ocular_config["projection_dim"]),
        hidden_size=int(ocular_config["hidden_size"]),
        layers=int(ocular_config["layers"]),
        dropout=float(ocular_config["dropout"]),
    )
    ocular.load_state_dict(ocular_payload["model_state"])
    gate_ocular = None
    if gate_ocular_payload is not None:
        gate_ocular_config = gate_ocular_payload["model_config"]
        if int(gate_ocular_config.get("input_dim", 0)) != 2 * region_dim + 5:
            raise ValueError(
                "microsleep-gate ocular checkpoint input dimension mismatch"
            )
        gate_ocular = CausalOcularLSTM(
            input_dim=int(gate_ocular_config["input_dim"]),
            projection_dim=int(gate_ocular_config["projection_dim"]),
            hidden_size=int(gate_ocular_config["hidden_size"]),
            layers=int(gate_ocular_config["layers"]),
            dropout=float(gate_ocular_config["dropout"]),
        )
        gate_ocular.load_state_dict(gate_ocular_payload["model_state"])
    temporal_config = temporal_payload["model_config"]
    expected_temporal_input = (
        visual_embedding_dim
        + (len(PHASE_NAMES) if phase_aware else 0)
        + len(EVIDENCE_FEATURE_NAMES)
        + 1
    )
    if int(temporal_config.get("input_dim", 0)) != expected_temporal_input:
        raise ValueError("temporal checkpoint input dimension mismatch")
    temporal = CausalFourStateLSTM(
        input_dim=int(temporal_config["input_dim"]),
        projection_dim=int(temporal_config["projection_dim"]),
        hidden_size=int(temporal_config["hidden_size"]),
        layers=int(temporal_config["layers"]),
        dropout=float(temporal_config["dropout"]),
        input_dropout=float(temporal_config["input_dropout"]),
    )
    temporal.load_state_dict(temporal_payload["model_state"])
    image_size = tuple(int(value) for value in visual_config["image_size"])
    runtime = MobileNetLSTMRuntime(
        encoder=encoder,
        ocular=ocular,
        gate_ocular=gate_ocular,
        temporal=temporal,
        visual_embedding_dim=visual_embedding_dim,
        embedding_dim=embedding_dim,
        region_dim=region_dim,
        fps=fps,
        image_size=image_size,
        device=device,
        uncertain_eye_gap_seconds=float(
            evidence_config.get(
                "uncertain_gap_seconds",
                0.1,
            )
        ),
        ocular_phase_features=phase_aware,
        phase_calibrated_closure=causal_drowsy,
        causal_drowsy_gate=causal_drowsy,
        causal_fatigue_episode=fatigue_episode,
        fatigue_episode_seconds=(
            fatigue_episode_seconds if fatigue_episode else 15.0
        ),
        fatigue_episode_yawn_block_threshold=fatigue_yawn_block_threshold,
        fatigue_episode_distraction_block_threshold=(
            fatigue_distraction_block_threshold
        ),
    )

    ultra = UltraLightFaceDetector(ultralight_model)
    tracked = TrackedFaceDetector(ultra, max_missed_frames=3)
    periodic = PeriodicFaceDetector(
        tracked,
        interval_frames=detector_interval,
    )
    landmarker = MediaPipeFaceLandmarker(
        landmark_model,
        min_detection_confidence=0.3,
    )
    return LSTMDemoPipeline(
        runtime=runtime,
        region_detector=FaceBoxLandmarkDetector(
            periodic,
            landmarker,
            padding=0.15,
        ),
    )


def create_pipeline(
    *,
    eye_checkpoint: Path,
    primitive_checkpoint: Path,
    ultralight_model: Path,
    landmark_model: Path,
    device: torch.device,
    fps: float,
    detector_interval: int,
    primitive_interval: int = 2,
    mouth_checkpoint: Path | None = None,
    cabin_temporal_checkpoint: Path | None = None,
    face_mouth_temporal_checkpoint: Path | None = None,
    five_state_checkpoint: Path | None = None,
) -> tuple[RealtimeDriverPipeline, MediaPipeFaceLandmarker]:
    specialist_paths = (cabin_temporal_checkpoint, face_mouth_temporal_checkpoint)
    if any(path is not None for path in specialist_paths) and not all(
        path is not None for path in specialist_paths
    ):
        raise ValueError(
            "cabin temporal and face-mouth temporal checkpoints must be supplied together"
        )
    if five_state_checkpoint is not None and any(
        path is not None for path in specialist_paths
    ):
        raise ValueError("choose either unified five-state or specialist temporal checkpoints")
    if (
        five_state_checkpoint is not None
        or any(path is not None for path in specialist_paths)
    ) and mouth_checkpoint is None:
        raise ValueError("temporal runtime requires the mouth checkpoint")
    eye_payload = torch.load(eye_checkpoint, map_location=device, weights_only=False)
    eye_config = EyeTemporalConfig(**eye_payload["config"])
    eye_model = EyeTemporalNet(
        embedding_dim=eye_config.embedding_dim,
        temporal_channels=eye_config.tcn_channels,
        dilations=eye_config.tcn_dilations,
        visibility_floor=eye_config.visibility_floor,
    )
    eye_model.load_state_dict(eye_payload["model_state"])

    primitive_payload = torch.load(
        primitive_checkpoint, map_location=device, weights_only=False
    )
    primitive_config = primitive_payload["model_config"]
    primitive_model = SpatialPrimitiveNet(
        pretrained=False,
        embedding_dim=int(primitive_config["embedding_dim"]),
    )
    primitive_model.load_state_dict(primitive_payload["model_state"])

    mouth_model = None
    cabin_temporal_model = None
    face_mouth_temporal_model = None
    cabin_calibration = None
    face_mouth_calibration = None
    unified_model = None
    unified_sequence_length = 100
    expected_split = _normalized_split(primitive_payload)
    mouth_payload = None
    if mouth_checkpoint is not None:
        assert mouth_checkpoint is not None
        mouth_payload = torch.load(
            mouth_checkpoint, map_location=device, weights_only=False
        )
        if not mouth_split_compatible(
            _normalized_split(mouth_payload), expected_split
        ):
            raise ValueError("mouth checkpoint split mismatch")
        mouth_config = mouth_payload.get("model_config", {})
        mouth_dim = int(mouth_config.get("embedding_dim", 0))
        if mouth_dim != 64:
            raise ValueError(
                f"runtime temporal schema requires 64-D mouth embeddings, got {mouth_dim}"
            )
        mouth_model = MouthVisualNet(embedding_dim=mouth_dim)
        mouth_model.load_state_dict(mouth_payload["model_state"])

    if all(path is not None for path in specialist_paths):
        assert cabin_temporal_checkpoint is not None
        assert face_mouth_temporal_checkpoint is not None
        assert mouth_checkpoint is not None
        assert mouth_payload is not None
        cabin_payload = torch.load(
            cabin_temporal_checkpoint, map_location=device, weights_only=False
        )
        face_mouth_payload = torch.load(
            face_mouth_temporal_checkpoint, map_location=device, weights_only=False
        )
        for name, payload in (
            ("cabin temporal", cabin_payload),
            ("face-mouth temporal", face_mouth_payload),
        ):
            if _normalized_split(payload) != expected_split:
                raise ValueError(f"{name} checkpoint split mismatch")
        expected_fingerprints = {
            "spatial": checkpoint_fingerprint(primitive_checkpoint),
            "mouth": checkpoint_fingerprint(mouth_checkpoint),
        }
        for name, payload, specialist in (
            ("cabin temporal", cabin_payload, "cabin"),
            ("face-mouth temporal", face_mouth_payload, "face_mouth"),
        ):
            if payload.get("schema_version") != 1 or payload.get("model_kind") != "tcn":
                raise ValueError(f"{name} checkpoint is not a schema-1 TCN")
            if payload.get("specialist") != specialist:
                raise ValueError(f"{name} checkpoint specialist mismatch")
            if payload.get("source_fingerprints") != expected_fingerprints:
                raise ValueError(f"{name} checkpoint source fingerprint mismatch")

        cabin_temporal_model = CabinEmbeddingTCN(
            input_dim=int(cabin_payload["input_dim"]),
            channels=int(cabin_payload["channels"]),
        )
        cabin_temporal_model.load_state_dict(cabin_payload["model_state"])
        face_mouth_temporal_model = FaceMouthEmbeddingTCN(
            input_dim=int(face_mouth_payload["input_dim"]),
            channels=int(face_mouth_payload["channels"]),
        )
        face_mouth_temporal_model.load_state_dict(face_mouth_payload["model_state"])
        cabin_calibration = cabin_payload["calibration"]
        face_mouth_calibration = face_mouth_payload["calibration"]

    if five_state_checkpoint is not None:
        assert mouth_checkpoint is not None
        if not mouth_split_compatible(_normalized_split(eye_payload), expected_split):
            raise ValueError("eye checkpoint split mismatch")
        payload = torch.load(
            five_state_checkpoint, map_location=device, weights_only=False
        )
        if payload.get("schema_version") != 1 or payload.get("model_kind") != "tcn":
            raise ValueError("five-state checkpoint is not a schema-1 causal TCN")
        if _normalized_split(payload) != expected_split:
            raise ValueError("five-state checkpoint split mismatch")
        expected_fingerprints = {
            "spatial": checkpoint_fingerprint(primitive_checkpoint),
            "mouth": checkpoint_fingerprint(mouth_checkpoint),
            "eye": checkpoint_fingerprint(eye_checkpoint),
        }
        if payload.get("source_fingerprints") != expected_fingerprints:
            raise ValueError("five-state checkpoint source fingerprint mismatch")
        config = payload.get("model_config", {})
        if int(config.get("input_dim", 0)) != 718:
            raise ValueError("five-state checkpoint must use the 718-D fusion schema")
        unified_model = FiveStateFusionTCN(
            input_dim=718,
            channels=int(config["channels"]),
            dilations=tuple(int(value) for value in config["dilations"]),
        )
        unified_model.load_state_dict(payload["model_state"])
        unified_sequence_length = int(config["sequence_length"])

    ultra = UltraLightFaceDetector(ultralight_model)
    tracked = TrackedFaceDetector(ultra, max_missed_frames=3)
    landmarker = MediaPipeFaceLandmarker(landmark_model, min_detection_confidence=0.3)
    synchronous_detector = FaceBoxLandmarkDetector(tracked, landmarker, padding=0.15)
    detector = AsyncFaceLandmarkDetector(
        synchronous_detector, interval_frames=detector_interval
    )
    return (
        RealtimeDriverPipeline(
            eye_model=eye_model,
            primitive_model=primitive_model,
            mouth_model=mouth_model,
            cabin_temporal_model=cabin_temporal_model,
            face_mouth_temporal_model=face_mouth_temporal_model,
            unified_model=unified_model,
            cabin_calibration=cabin_calibration,
            face_mouth_calibration=face_mouth_calibration,
            landmark_detector=detector,
            device=device,
            fps=fps,
            sequence_length=eye_config.sequence_length,
            primitive_interval_frames=primitive_interval,
            unified_sequence_length=unified_sequence_length,
        ),
        landmarker,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-time exclusive five-state DMD demo")
    parser.add_argument("--lstm-visual-checkpoint", type=Path)
    parser.add_argument("--lstm-ocular-checkpoint", type=Path)
    parser.add_argument("--lstm-gate-ocular-checkpoint", type=Path)
    parser.add_argument("--lstm-temporal-checkpoint", type=Path)
    parser.add_argument("--eye-checkpoint", type=Path)
    parser.add_argument("--primitive-checkpoint", type=Path)
    parser.add_argument("--mouth-checkpoint", type=Path)
    parser.add_argument("--cabin-temporal-checkpoint", type=Path)
    parser.add_argument("--face-mouth-temporal-checkpoint", type=Path)
    parser.add_argument("--five-state-checkpoint", type=Path)
    parser.add_argument("--ultralight-model", type=Path, required=True)
    parser.add_argument("--landmark-model", type=Path, required=True)
    parser.add_argument("--source", default="0", help="camera index or video path")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--detector-interval", type=int, default=5)
    parser.add_argument("--primitive-interval", type=int, default=2)
    parser.add_argument("--display", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import cv2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_paths = (
        args.lstm_visual_checkpoint,
        args.lstm_ocular_checkpoint,
        args.lstm_temporal_checkpoint,
    )
    legacy_paths = (
        args.eye_checkpoint,
        args.primitive_checkpoint,
        args.mouth_checkpoint,
        args.cabin_temporal_checkpoint,
        args.face_mouth_temporal_checkpoint,
        args.five_state_checkpoint,
    )
    if any(path is not None for path in lstm_paths) and not all(
        path is not None for path in lstm_paths
    ):
        raise ValueError(
            "visual, ocular, and temporal LSTM checkpoints must be supplied together"
        )
    if args.lstm_gate_ocular_checkpoint is not None and not all(
        path is not None for path in lstm_paths
    ):
        raise ValueError(
            "the microsleep-gate ocular checkpoint requires all three LSTM "
            "checkpoints"
        )
    lstm_mode = all(path is not None for path in lstm_paths)
    if lstm_mode and any(path is not None for path in legacy_paths):
        raise ValueError(
            "the LSTM pipeline cannot be combined with legacy checkpoints"
        )
    if not lstm_mode and (
        args.eye_checkpoint is None or args.primitive_checkpoint is None
    ):
        raise ValueError(
            "supply both LSTM checkpoints or both legacy eye and primitive checkpoints"
        )

    legacy_landmarker: MediaPipeFaceLandmarker | None = None
    if lstm_mode:
        assert args.lstm_visual_checkpoint is not None
        assert args.lstm_ocular_checkpoint is not None
        assert args.lstm_temporal_checkpoint is not None
        pipeline = create_lstm_pipeline(
            visual_checkpoint=args.lstm_visual_checkpoint,
            ocular_checkpoint=args.lstm_ocular_checkpoint,
            temporal_checkpoint=args.lstm_temporal_checkpoint,
            ultralight_model=args.ultralight_model,
            landmark_model=args.landmark_model,
            device=device,
            fps=args.fps,
            detector_interval=args.detector_interval,
            gate_ocular_checkpoint=args.lstm_gate_ocular_checkpoint,
        )
    else:
        assert args.eye_checkpoint is not None
        assert args.primitive_checkpoint is not None
        pipeline, legacy_landmarker = create_pipeline(
            eye_checkpoint=args.eye_checkpoint,
            primitive_checkpoint=args.primitive_checkpoint,
            ultralight_model=args.ultralight_model,
            landmark_model=args.landmark_model,
            device=device,
            fps=args.fps,
            detector_interval=args.detector_interval,
            primitive_interval=args.primitive_interval,
            mouth_checkpoint=args.mouth_checkpoint,
            cabin_temporal_checkpoint=args.cabin_temporal_checkpoint,
            face_mouth_temporal_checkpoint=args.face_mouth_temporal_checkpoint,
            five_state_checkpoint=args.five_state_checkpoint,
        )
    source: int | str = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        pipeline.close()
        if legacy_landmarker is not None:
            legacy_landmarker.close()
        raise SystemExit(f"cannot open video source: {args.source}")
    frame_index = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            prediction = pipeline.process(
                Image.fromarray(rgb), timestamp=frame_index / args.fps
            )
            if lstm_mode:
                assert isinstance(prediction, RuntimeFiveStatePrediction)
                output = {
                    "frame": frame_index,
                    "state_id": prediction.state_id,
                    "state": FiveState(prediction.state_id).name.lower(),
                    "confidence": max(prediction.probabilities),
                    "probabilities": prediction.probabilities,
                    "diagnostics": {
                        "closure_duration_seconds": (
                            prediction.closure_duration_seconds
                        ),
                        "perclos": prediction.perclos,
                        "slow_perclos": prediction.slow_perclos,
                        "perclos_reliable": prediction.perclos_reliable,
                        "nod_probability": prediction.nod_probability,
                    },
                }
            else:
                output = {
                    "frame": frame_index,
                    "state_id": prediction.result.state_id,
                    "state": prediction.result.state.value,
                    "confidence": prediction.result.confidence,
                    "visible": prediction.result.visible,
                    "eye_phase": prediction.eye_phase,
                    "diagnostics": prediction.result.diagnostics,
                }
            print(json.dumps(output), flush=True)
            if args.display:
                cv2.putText(
                    bgr,
                    f"{output['state_id']} {output['state']}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("DMD driver state", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_index += 1
    finally:
        capture.release()
        if args.display:
            cv2.destroyAllWindows()
        pipeline.close()
        if legacy_landmarker is not None:
            legacy_landmarker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
