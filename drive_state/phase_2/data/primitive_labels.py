"""Protocol-aware DMD primitive vocabularies and partial-label encoding."""

from __future__ import annotations

from typing import Mapping


IGNORE_INDEX = -100

DRIVER_ACTIONS = (
    "safe_drive",
    "phonecall_left",
    "phonecall_right",
    "reach_side",
    "texting_right",
    "texting_left",
    "hair_and_makeup",
    "radio",
    "drinking",
    "talking_to_passenger",
    "reach_backseat",
    "standstill_or_waiting",
    "change_gear",
)
DISTRACTING_ACTIONS = frozenset(
    {
        "phonecall_left",
        "phonecall_right",
        "reach_side",
        "texting_right",
        "texting_left",
        "hair_and_makeup",
        "radio",
        "drinking",
        "talking_to_passenger",
        "reach_backseat",
    }
)
HANDS_USING_WHEEL = ("both", "only_left", "only_right", "none")
YAWN_CLASSES = ("none", "with_hand", "without_hand")
GAZE_ZONES = (
    "front",
    "steering_wheel",
    "left_mirror",
    "front_right",
    "infotainment",
    "right_mirror",
    "center_mirror",
    "right",
    "left",
)
HANDS_ON_WHEEL = ("both_hands", "only_left", "only_right", "none")

TASK_CLASS_COUNTS = {
    "driver_action": len(DRIVER_ACTIONS),
    "distraction": 2,
    "road_gaze": 2,
    "hands_using_wheel": len(HANDS_USING_WHEEL),
    "talking": 2,
    "yawn": len(YAWN_CLASSES),
    "blink": 2,
    "gaze_zone": len(GAZE_ZONES),
    "hands_on_wheel": len(HANDS_ON_WHEEL),
    "moving_hands": 2,
}


def _normalized(row: Mapping[str, str], field: str) -> str | None:
    if field not in row:
        return None
    return (row.get(field) or "").strip().lower()


def _index(value: str | None, classes: tuple[str, ...]) -> int:
    if value is None or value not in classes:
        return IGNORE_INDEX
    return classes.index(value)


def encode_primitive_row(protocol: str, row: Mapping[str, str]) -> dict[str, int]:
    """Encode one CSV row without turning absent task labels into negatives."""
    targets = {task: IGNORE_INDEX for task in TASK_CLASS_COUNTS}
    if protocol in {"s1", "s2", "s3"}:
        action = _normalized(row, "driver_actions")
        targets["driver_action"] = _index(action, DRIVER_ACTIONS)
        if targets["driver_action"] != IGNORE_INDEX:
            targets["distraction"] = int(action in DISTRACTING_ACTIONS)

        road_gaze = _normalized(row, "gaze_on_road")
        if road_gaze in {"looking_road", "not_looking_road"}:
            targets["road_gaze"] = int(road_gaze == "not_looking_road")
        targets["hands_using_wheel"] = _index(
            _normalized(row, "hands_using_wheel"), HANDS_USING_WHEEL
        )
        talking = _normalized(row, "talking")
        if talking is not None:
            targets["talking"] = int(talking == "talking") if talking in {"", "talking"} else IGNORE_INDEX

    if protocol == "s5":
        yawn = _normalized(row, "yawning")
        if yawn is not None:
            yawn_map = {
                "": 0,
                "yawning with hand": 1,
                "yawning without hand": 2,
            }
            targets["yawn"] = yawn_map.get(yawn, IGNORE_INDEX)

    if protocol in {"s5", "s6"}:
        blink = _normalized(row, "blinks")
        if blink is not None:
            targets["blink"] = int(blink == "blinking") if blink in {"", "blinking"} else IGNORE_INDEX

    if protocol == "s6":
        targets["gaze_zone"] = _index(_normalized(row, "gaze_zone"), GAZE_ZONES)
        targets["hands_on_wheel"] = _index(
            _normalized(row, "hands_on_wheel"), HANDS_ON_WHEEL
        )
        moving = _normalized(row, "moving_hands")
        if moving in {"not_moving", "moving"}:
            targets["moving_hands"] = int(moving == "moving")
    return targets
