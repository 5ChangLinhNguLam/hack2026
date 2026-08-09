"""Memmapped eye-crop storage and PyTorch temporal clip dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .eye_sequences import (
    IGNORE_INDEX,
    EyeClipRef,
    EyeSequence,
    eye_auxiliary_targets,
)
from .eye_crop_cache import EyeCropStore, save_eye_crop_cache

__all__ = ["EyeClipDataset", "EyeCropStore", "save_eye_crop_cache"]


class EyeClipDataset(Dataset[dict[str, Tensor | str | int]]):
    """Return normalized causal clips and all masks needed by the eye loss."""

    def __init__(
        self,
        sequences: Sequence[EyeSequence],
        references: Sequence[EyeClipRef],
        *,
        cache_dir: Path | str,
        sequence_length: int,
    ) -> None:
        self.sequences = tuple(sequences)
        self.references = tuple(references)
        self.sequence_length = sequence_length
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.stores = {
            sequence.session_name: EyeCropStore(cache_dir, sequence.session_name)
            for sequence in self.sequences
        }
        for sequence in self.sequences:
            self.stores[sequence.session_name].rows_for(sequence.frame_ids)

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int) -> dict[str, Tensor | str | int]:
        reference = self.references[index]
        sequence = self.sequences[reference.sequence_index]
        stop = reference.start + self.sequence_length
        if reference.start < 0 or stop > len(sequence):
            raise IndexError(f"clip {reference} exceeds session {sequence.session_name}")

        frame_ids = sequence.frame_ids[reference.start:stop]
        store = self.stores[sequence.session_name]
        rows = store.rows_for(frame_ids)
        # Advanced indexing makes an owned copy, safe for torch even though the
        # backing full-session array is a read-only mmap.
        crops = np.asarray(store.eyes[rows]).copy()
        eyes = torch.from_numpy(crops).permute(0, 1, 4, 2, 3).float()
        eyes = eyes.div_(127.5).sub_(1.0)
        visibility = torch.from_numpy(store.visibility[rows].copy())

        phase_values = sequence.phase_targets[reference.start:stop]
        closedness, closedness_mask, moving = eye_auxiliary_targets(phase_values)
        phase = torch.tensor(phase_values, dtype=torch.long)
        valid_mask = phase.ne(IGNORE_INDEX) & visibility.gt(0.0).any(dim=1)
        return {
            "eyes": eyes,
            "visibility": visibility,
            "phase": phase,
            "valid_mask": valid_mask,
            "closedness": torch.tensor(closedness, dtype=torch.float32),
            "closedness_mask": torch.tensor(closedness_mask, dtype=torch.bool),
            "moving": torch.tensor(moving, dtype=torch.float32),
            "session": sequence.session_name,
            "subject": sequence.subject_id,
            "start": reference.start,
        }
