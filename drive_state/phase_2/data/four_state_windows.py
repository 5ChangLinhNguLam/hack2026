"""Four-state temporal windows joined to denoised ocular evidence."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .five_state_events import StateTargetRef
from .five_state_labels import IGNORE_INDEX
from .five_state_standardization import StandardizedSession
from .five_state_visual_cache import VisualSessionCache
from .four_state_labels import five_to_four_target
from .ocular_cache import OcularSessionCache


class FourStateWindowDataset(
    Dataset[dict[str, Tensor | str | int | float]]
):
    """Pair immutable visual and ocular caches for causal state learning."""

    def __init__(
        self,
        visual_caches: Sequence[VisualSessionCache],
        ocular_caches: Sequence[OcularSessionCache],
        sessions: Sequence[StandardizedSession],
        references: Sequence[StateTargetRef],
        *,
        length: int,
        gate_ocular_caches: Sequence[OcularSessionCache] | None = None,
    ) -> None:
        if length <= 0:
            raise ValueError("four-state window length must be positive")
        if not (
            len(visual_caches) == len(ocular_caches) == len(sessions)
        ):
            raise ValueError(
                "visual, ocular, and standardized sessions must align"
            )
        self.visual_caches = tuple(visual_caches)
        self.ocular_caches = tuple(ocular_caches)
        self.gate_ocular_caches = (
            self.ocular_caches
            if gate_ocular_caches is None
            else tuple(gate_ocular_caches)
        )
        if len(self.gate_ocular_caches) != len(self.ocular_caches):
            raise ValueError("microsleep-gate ocular caches must align")
        self.sessions = tuple(sessions)
        self.references = tuple(references)
        self.length = int(length)
        for visual, ocular, gate_ocular, session in zip(
            self.visual_caches,
            self.ocular_caches,
            self.gate_ocular_caches,
            self.sessions,
            strict=True,
        ):
            identities = {
                (visual.session, visual.subject, visual.protocol),
                (ocular.session, ocular.subject, ocular.protocol),
                (session.session, session.subject, session.protocol),
            }
            if len(identities) != 1:
                raise ValueError("visual/ocular cache session identity mismatch")
            if not (
                math.isclose(visual.fps, ocular.fps)
                and math.isclose(visual.fps, session.fps)
            ):
                raise ValueError("visual/ocular cache session FPS mismatch")
            expected_frames = np.asarray(session.frame_ids, dtype=np.int32)
            if not np.array_equal(visual.frame_ids, expected_frames) or not np.array_equal(
                ocular.frame_ids,
                expected_frames,
            ):
                raise ValueError("visual/ocular cache frame IDs do not align")
            if (
                ocular.visual_checkpoint_fingerprint
                != visual.visual_checkpoint_fingerprint
                or ocular.region_cache_fingerprint
                != visual.region_cache_fingerprint
            ):
                raise ValueError("visual/ocular cache fingerprint mismatch")
            if dict(ocular.split) != dict(visual.split):
                raise ValueError("visual/ocular cache split mismatch")
            if (
                (gate_ocular.session, gate_ocular.subject, gate_ocular.protocol)
                != (visual.session, visual.subject, visual.protocol)
                or not math.isclose(gate_ocular.fps, visual.fps)
                or not np.array_equal(gate_ocular.frame_ids, expected_frames)
            ):
                raise ValueError("microsleep-gate ocular cache does not align")
            if (
                gate_ocular.visual_checkpoint_fingerprint
                != visual.visual_checkpoint_fingerprint
                or gate_ocular.region_cache_fingerprint
                != visual.region_cache_fingerprint
                or dict(gate_ocular.split) != dict(visual.split)
            ):
                raise ValueError(
                    "microsleep-gate ocular cache fingerprint mismatch"
                )
        for reference in self.references:
            if not 0 <= reference.session_index < len(self.sessions):
                raise IndexError("four-state target has invalid session index")
            session = self.sessions[reference.session_index]
            if not 0 <= reference.target_index < len(session.frame_ids):
                raise IndexError("four-state target has invalid frame index")
            if session.targets.event_ids[reference.target_index] != reference.event_id:
                raise ValueError("four-state target event ID mismatch")

    def __len__(self) -> int:
        return len(self.references)

    @staticmethod
    def _history(
        values: np.ndarray,
        *,
        start: int,
        stop: int,
        pad: int,
    ) -> Tensor:
        tensor = torch.from_numpy(
            np.asarray(values[start:stop], dtype=np.float32).copy()
        )
        if pad:
            tensor = torch.cat(
                (
                    torch.zeros(
                        (pad, *tensor.shape[1:]),
                        dtype=torch.float32,
                    ),
                    tensor,
                )
            )
        return tensor

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Tensor | str | int | float]:
        reference = self.references[index]
        visual = self.visual_caches[reference.session_index]
        ocular = self.ocular_caches[reference.session_index]
        gate_ocular = self.gate_ocular_caches[reference.session_index]
        session = self.sessions[reference.session_index]
        target = reference.target_index
        start = max(0, target - self.length + 1)
        stop = target + 1
        pad = self.length - (stop - start)
        five_targets = torch.tensor(
            session.targets.targets[start:stop],
            dtype=torch.long,
        )
        four_targets = torch.tensor(
            [five_to_four_target(value) for value in five_targets.tolist()],
            dtype=torch.long,
        )
        confidence = torch.tensor(
            session.targets.confidence[start:stop],
            dtype=torch.float32,
        )
        if pad:
            ignored = torch.full((pad,), IGNORE_INDEX, dtype=torch.long)
            five_targets = torch.cat((ignored, five_targets))
            four_targets = torch.cat((ignored.clone(), four_targets))
            confidence = torch.cat(
                (torch.zeros(pad, dtype=torch.float32), confidence)
            )
        five_target = int(session.targets.targets[target])
        return {
            "visual_embedding": self._history(
                visual.visual_embedding,
                start=start,
                stop=stop,
                pad=pad,
            ),
            "evidence": self._history(
                ocular.evidence,
                start=start,
                stop=stop,
                pad=pad,
            ),
            "ocular_phase_probabilities": self._history(
                ocular.phase_probabilities,
                start=start,
                stop=stop,
                pad=pad,
            ),
            "context_valid": torch.cat(
                (
                    torch.zeros(pad, dtype=torch.float32),
                    torch.ones(stop - start, dtype=torch.float32),
                )
            ),
            "four_state_targets": four_targets,
            "five_state_targets": five_targets,
            "confidence_sequence": confidence,
            "four_state_target": torch.tensor(
                five_to_four_target(five_target),
                dtype=torch.long,
            ),
            "five_state_target": torch.tensor(five_target, dtype=torch.long),
            "confidence": torch.tensor(
                session.targets.confidence[target],
                dtype=torch.float32,
            ),
            "drowsy_active": torch.tensor(
                bool(ocular.drowsy_active[target]),
                dtype=torch.bool,
            ),
            "microsleep_active": torch.tensor(
                bool(gate_ocular.microsleep_active[target]),
                dtype=torch.bool,
            ),
            "session": session.session,
            "event_id": reference.event_id,
            "target_frame_id": int(session.frame_ids[target]),
            "target_timestamp": target / session.fps,
        }


__all__ = ["FourStateWindowDataset"]
