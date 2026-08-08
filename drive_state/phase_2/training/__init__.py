"""Training loops and checkpoint contracts."""

from .eye_trainer import (
    SubjectSplitMetadata,
    compute_phase_class_weights,
    evaluate_eye,
    load_eye_checkpoint,
    save_eye_checkpoint,
    train_eye_epoch,
)
from .primitive_trainer import (
    evaluate_primitives,
    save_primitive_checkpoint,
    train_primitive_epoch,
)
from .five_state_trainer import (
    evaluate_five_state,
    five_state_metrics,
    load_five_state_checkpoint,
    save_five_state_checkpoint,
    train_five_state_epoch,
)

__all__ = [
    "SubjectSplitMetadata",
    "compute_phase_class_weights",
    "evaluate_eye",
    "load_eye_checkpoint",
    "save_eye_checkpoint",
    "train_eye_epoch",
    "evaluate_primitives",
    "save_primitive_checkpoint",
    "train_primitive_epoch",
    "evaluate_five_state",
    "five_state_metrics",
    "load_five_state_checkpoint",
    "save_five_state_checkpoint",
    "train_five_state_epoch",
]
