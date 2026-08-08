"""Stateful real-time inference for one MobileNetV3-Large plus causal LSTM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from PIL import Image
import torch
from torch import nn

from .data.evidence_regions import project_normalized_boxes_to_letterbox
from .data.five_state_visual_cache import EYE_EVIDENCE_CONFIG
from .data.four_state_labels import compose_five_state_probabilities
from .data.primitive_dataset import letterbox, normalized_image_tensor
from .models.mobilenet_lstm import CausalFourStateLSTM
from .models.ocular_lstm import (
    CausalOcularLSTM,
    assemble_ocular_features,
    phase_calibrated_closed_probability,
)
from .state.causal_evidence import (
    CausalEvidenceAccumulator,
    PrimitiveFrameEvidence,
)
from .state.fatigue_episode import CausalFatigueEpisode


@dataclass(frozen=True)
class RuntimeFiveStatePrediction:
    state_id: int
    probabilities: tuple[float, ...]
    closure_duration_seconds: float
    perclos: tuple[float, float, float]
    slow_perclos: tuple[float, float, float]
    perclos_reliable: tuple[bool, bool, bool]
    nod_probability: float
    microsleep_active: bool
    ocular_phase_probabilities: tuple[float, float, float, float]
    closed_probability: float
    ocular_reliability: float
    drowsy_episode_active: bool = False


class MobileNetLSTMRuntime:
    """Maintain the canonical evidence accumulator and LSTM hidden state."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        ocular: CausalOcularLSTM,
        gate_ocular: CausalOcularLSTM | None = None,
        temporal: CausalFourStateLSTM,
        visual_embedding_dim: int,
        embedding_dim: int,
        region_dim: int,
        fps: float,
        image_size: tuple[int, int],
        device: torch.device,
        perclos_windows: Sequence[float] = (10.0, 30.0, 60.0),
        uncertain_eye_gap_seconds: float = 0.1,
        amp: bool = True,
        ocular_phase_features: bool = False,
        phase_calibrated_closure: bool = False,
        causal_drowsy_gate: bool = False,
        causal_fatigue_episode: bool = False,
        fatigue_episode_seconds: float = 15.0,
        fatigue_episode_yawn_block_threshold: float = 0.7,
        fatigue_episode_distraction_block_threshold: float = 0.7,
    ) -> None:
        windows = tuple(sorted({float(value) for value in perclos_windows}))
        if min(visual_embedding_dim, embedding_dim, region_dim) <= 0 or fps <= 0.0:
            raise ValueError(
                "runtime embedding dimension and FPS must be positive"
            )
        if visual_embedding_dim != embedding_dim + 4 * region_dim:
            raise ValueError("runtime visual embedding dimensions do not compose")
        if uncertain_eye_gap_seconds < 0.0:
            raise ValueError("uncertain eye gap must be non-negative")
        if causal_fatigue_episode and not causal_drowsy_gate:
            raise ValueError(
                "causal fatigue episodes require the causal drowsy gate"
            )
        episode_thresholds = (
            fatigue_episode_yawn_block_threshold,
            fatigue_episode_distraction_block_threshold,
        )
        if any(value < 0.0 or value > 1.0 for value in episode_thresholds):
            raise ValueError(
                "fatigue episode conflict thresholds must be in [0, 1]"
            )
        if any(value <= 0 for value in image_size) or len(windows) != 3:
            raise ValueError(
                "runtime image size and three PERCLOS windows are required"
            )
        self.encoder = encoder.to(device).eval()
        self.ocular = ocular.to(device).eval()
        self.gate_ocular = (
            None if gate_ocular is None else gate_ocular.to(device).eval()
        )
        self.temporal = temporal.to(device).eval()
        self.visual_embedding_dim = int(visual_embedding_dim)
        self.embedding_dim = int(embedding_dim)
        self.region_dim = int(region_dim)
        self.fps = float(fps)
        self.image_size = tuple(int(value) for value in image_size)
        self.device = device
        self.perclos_windows = windows
        self.uncertain_eye_gap_seconds = float(uncertain_eye_gap_seconds)
        self.amp = bool(amp)
        self.ocular_phase_features = bool(ocular_phase_features)
        self.phase_calibrated_closure = bool(phase_calibrated_closure)
        self.causal_drowsy_gate = bool(causal_drowsy_gate)
        self.causal_fatigue_episode = bool(causal_fatigue_episode)
        self.fatigue_episode_yawn_block_threshold = float(
            fatigue_episode_yawn_block_threshold
        )
        self.fatigue_episode_distraction_block_threshold = float(
            fatigue_episode_distraction_block_threshold
        )
        self.fatigue_episode = (
            CausalFatigueEpisode(
                fps=self.fps,
                hold_seconds=fatigue_episode_seconds,
            )
            if self.causal_fatigue_episode
            else None
        )
        self.ocular_hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self.gate_ocular_hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self.temporal_hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self.steps = 0
        if self.fatigue_episode is not None:
            self.fatigue_episode.reset()
        self.accumulator = self._new_accumulator(
            behavioral_gate_suppression=self.causal_drowsy_gate,
        )
        self.gate_accumulator = (
            None
            if self.gate_ocular is None
            else self._new_accumulator(behavioral_gate_suppression=False)
        )

    def _new_accumulator(
        self,
        *,
        behavioral_gate_suppression: bool,
    ) -> CausalEvidenceAccumulator:
        evidence_config = dict(EYE_EVIDENCE_CONFIG)
        evidence_config["uncertain_gap_seconds"] = (
            self.uncertain_eye_gap_seconds
        )
        return CausalEvidenceAccumulator(
            fps=self.fps,
            perclos_windows=self.perclos_windows,
            behavioral_gate_suppression=behavioral_gate_suppression,
            **evidence_config,
        )

    def reset(self) -> None:
        self.ocular_hidden = None
        self.gate_ocular_hidden = None
        self.temporal_hidden = None
        self.steps = 0
        if self.fatigue_episode is not None:
            self.fatigue_episode.reset()
        self.accumulator = self._new_accumulator(
            behavioral_gate_suppression=self.causal_drowsy_gate,
        )
        self.gate_accumulator = (
            None
            if self.gate_ocular is None
            else self._new_accumulator(behavioral_gate_suppression=False)
        )

    @torch.inference_mode()
    def process(
        self,
        image: Image.Image,
        *,
        region_boxes: Sequence[Sequence[float]],
        region_visibility: Sequence[float],
        timestamp: float | None = None,
    ) -> RuntimeFiveStatePrediction:
        rgb = image.convert("RGB")
        boxes = np.asarray(region_boxes, dtype=np.float32)
        visibility = np.asarray(region_visibility, dtype=np.float32)
        if boxes.shape != (4, 4):
            raise ValueError("runtime region boxes must have shape (4, 4)")
        if visibility.shape != (4,):
            raise ValueError("runtime region visibility must have shape (4,)")
        if np.any(~np.isfinite(boxes)) or np.any(
            (boxes < 0.0) | (boxes > 1.0)
        ):
            raise ValueError(
                "runtime region boxes must be normalized and finite"
            )
        if np.any(~np.isfinite(visibility)) or np.any(
            (visibility < 0.0) | (visibility > 1.0)
        ):
            raise ValueError("runtime region visibility must be in [0, 1]")
        projected = project_normalized_boxes_to_letterbox(
            boxes,
            source_size=rgb.size,
            target_size=self.image_size,
        )
        tensor = normalized_image_tensor(
            letterbox(rgb, self.image_size),
            self.image_size,
        ).unsqueeze(0).to(self.device)
        box_tensor = torch.from_numpy(projected).unsqueeze(0).to(self.device)
        visibility_tensor = (
            torch.from_numpy(visibility).unsqueeze(0).to(self.device)
        )
        with torch.autocast(
            device_type=self.device.type,
            enabled=self.amp and self.device.type == "cuda",
        ):
            visual = self.encoder(
                tensor,
                box_tensor,
                visibility_tensor,
            )
        embedding = visual.embedding.float()
        if embedding.shape != (1, self.visual_embedding_dim):
            raise ValueError("runtime encoder embedding dimension mismatch")
        if visual.region_embeddings.shape != (1, 4, self.region_dim):
            raise ValueError("runtime encoder region embedding dimension mismatch")
        eye_probabilities = visual.eye_logits.float().softmax(dim=-1)
        ocular_features = assemble_ocular_features(
            region_embeddings=visual.region_embeddings.float(),
            raw_eye_probabilities=eye_probabilities,
            eye_visibility=visibility_tensor[:, 1:3].float(),
        )
        ocular_output, self.ocular_hidden = self.ocular.step(
            ocular_features,
            self.ocular_hidden,
        )
        gate_output = None
        if self.gate_ocular is not None:
            gate_output, self.gate_ocular_hidden = self.gate_ocular.step(
                ocular_features,
                self.gate_ocular_hidden,
            )
        ocular_phase_probabilities = ocular_output.phase_logits.float().softmax(
            dim=-1
        )
        learned_closed_probability = ocular_output.closed_logits.float().sigmoid()
        closed_probability = (
            phase_calibrated_closed_probability(
                ocular_phase_probabilities,
                learned_closed_probability,
            )
            if self.phase_calibrated_closure
            else learned_closed_probability
        )
        learned_reliability = ocular_output.reliability_logits.float().sigmoid()
        observed_reliability = visibility_tensor[:, 1:3].float().mean(dim=-1)
        ocular_reliability = (learned_reliability * observed_reliability).clamp(
            0.0,
            1.0,
        )
        yawn_probability = float(
            visual.yawn_logits[0].float().softmax(dim=-1)[1]
        )
        distraction_probability = float(
            visual.distraction_logits[0].float().softmax(dim=-1)[1]
        )
        pose = visual.pose[0].float().mul(90.0).cpu()
        snapshot = self.accumulator.update(
            PrimitiveFrameEvidence(
                closed_probability=float(closed_probability[0].cpu()),
                eye_visibility=float(ocular_reliability[0].cpu()),
                head_pose=(float(pose[0]), float(pose[1])),
                yawn_probability=yawn_probability,
                mouth_visibility=float(visibility[3]),
                distraction_probability=distraction_probability,
                face_visibility=float(visibility[0]),
                timestamp=timestamp,
            )
        )
        gate_snapshot = snapshot
        if gate_output is not None:
            assert self.gate_accumulator is not None
            gate_closed_probability = gate_output.closed_logits.float().sigmoid()
            gate_learned_reliability = (
                gate_output.reliability_logits.float().sigmoid()
            )
            gate_reliability = (
                gate_learned_reliability * observed_reliability
            ).clamp(0.0, 1.0)
            gate_snapshot = self.gate_accumulator.update(
                PrimitiveFrameEvidence(
                    closed_probability=float(
                        gate_closed_probability[0].cpu()
                    ),
                    eye_visibility=float(gate_reliability[0].cpu()),
                    head_pose=(float(pose[0]), float(pose[1])),
                    yawn_probability=yawn_probability,
                    mouth_visibility=float(visibility[3]),
                    distraction_probability=distraction_probability,
                    face_visibility=float(visibility[0]),
                    timestamp=timestamp,
                )
            )
        drowsy_episode_active = False
        if self.fatigue_episode is not None:
            behavioral_conflict = bool(
                yawn_probability * float(visibility[3])
                >= self.fatigue_episode_yawn_block_threshold
                or distraction_probability
                >= self.fatigue_episode_distraction_block_threshold
            )
            drowsy_episode_active = self.fatigue_episode.update(
                event_active=snapshot.drowsy_active,
                blocked=behavioral_conflict,
                timestamp=timestamp,
            )
        feature_parts = [embedding[0]]
        if self.ocular_phase_features:
            feature_parts.append(ocular_phase_probabilities[0])
        feature_parts.extend(
            (
                torch.from_numpy(snapshot.vector).to(self.device),
                torch.ones(1, dtype=torch.float32, device=self.device),
            )
        )
        feature = torch.cat(tuple(feature_parts)).unsqueeze(0)
        logits, self.temporal_hidden = self.temporal.step(
            feature,
            self.temporal_hidden,
        )
        four_probabilities = logits.float().softmax(dim=-1)
        probabilities = compose_five_state_probabilities(
            four_probabilities,
            torch.tensor(
                [gate_snapshot.microsleep_active],
                dtype=torch.bool,
                device=self.device,
            ),
            drowsy_active=(
                torch.tensor(
                    [
                        drowsy_episode_active
                        if self.causal_fatigue_episode
                        else snapshot.drowsy_active
                    ],
                    dtype=torch.bool,
                    device=self.device,
                )
                if self.causal_drowsy_gate
                else None
            ),
        )[0].cpu()
        self.steps += 1
        return RuntimeFiveStatePrediction(
            state_id=int(probabilities.argmax()),
            probabilities=tuple(float(value) for value in probabilities),
            closure_duration_seconds=gate_snapshot.closure_duration_seconds,
            perclos=snapshot.perclos,
            slow_perclos=snapshot.slow_perclos,
            perclos_reliable=snapshot.perclos_reliable,
            nod_probability=snapshot.nod_probability,
            drowsy_episode_active=drowsy_episode_active,
            microsleep_active=gate_snapshot.microsleep_active,
            ocular_phase_probabilities=tuple(
                float(value)
                for value in ocular_phase_probabilities[0].cpu()
            ),
            closed_probability=float(closed_probability[0].cpu()),
            ocular_reliability=float(ocular_reliability[0].cpu()),
        )


__all__ = [
    "MobileNetLSTMRuntime",
    "RuntimeFiveStatePrediction",
]
