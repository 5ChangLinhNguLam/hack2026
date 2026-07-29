"""The Challenge 2 output vocabulary and its decomposition.

The organiser labels every frame with a `driver` block holding four fields::

    {"state": ..., "eye_state": ..., "head_pose": ..., "mouth_state": ...}

Across all 3,600 frames of the six Practice trips the triple
``(eye_state, head_pose, mouth_state)`` determines ``state`` exactly — five
triples, five states, no collisions. So the classifier does not have to learn
a 5-way decision directly: it estimates the three sub-signals from face
landmarks and reads the state off :data:`SIGNAL_TABLE`.
"""

from __future__ import annotations

# Exactly the strings the official scorer compares against. Lowercase, no
# aliases -- `evaluation.py` does a plain string equality test.
DRIVER_STATE_CLASSES: tuple[str, ...] = (
    "alert",
    "drowsy",
    "yawning",
    "distracted",
    "microsleep",
)

EYE_STATES: tuple[str, ...] = ("open", "partial", "closed")
HEAD_POSES: tuple[str, ...] = ("normal", "side", "down")
MOUTH_STATES: tuple[str, ...] = ("normal", "yawning")

#: (eye_state, head_pose, mouth_state) -> driver state. Derived from the
#: Practice labels; every one of the five entries is attested by 600-900 frames.
SIGNAL_TABLE: dict[tuple[str, str, str], str] = {
    ("open", "normal", "normal"): "alert",
    ("open", "side", "normal"): "distracted",
    ("partial", "down", "normal"): "drowsy",
    ("partial", "normal", "yawning"): "yawning",
    ("closed", "down", "normal"): "microsleep",
}


def state_from_signals(eye_state: str, head_pose: str, mouth_state: str) -> str:
    """Map the three sub-signals onto one of :data:`DRIVER_STATE_CLASSES`.

    Only five of the eighteen possible triples appear in the labels, so the
    estimator will regularly produce combinations the table has no entry for
    (``closed`` eyes with a ``normal`` head, say). Those fall through to a
    priority order rather than an error, because the scorer needs a state for
    every frame: the two states that carry the most risk win first, then
    distraction, then alert as the default.
    """
    key = (eye_state, head_pose, mouth_state)
    if key in SIGNAL_TABLE:
        return SIGNAL_TABLE[key]

    if mouth_state == "yawning":
        return "yawning"
    if eye_state == "closed":
        return "microsleep"
    if eye_state == "partial":
        return "drowsy"
    if head_pose in {"side", "down"}:
        return "distracted"
    return "alert"
