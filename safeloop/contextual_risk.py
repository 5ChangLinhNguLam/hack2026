"""Instantaneous product risk policy combining C1 and C2 inference.

This is a SafeLoop product decision, not the Challenge 3 scoring formula.
The explicit separation prevents a higher-is-dangerous risk value from being
mistaken for the higher-is-safer trip score.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


POLICY_VERSION = "safeloop-context-risk-v1"


@dataclass(frozen=True)
class ContextualRiskDecision:
    score_pct: float
    level: str
    action: str
    brake_request_pct: float
    reasons: tuple[str, ...]

    def diagnostic_row(self) -> dict[str, object]:
        return {
            "contextual_risk_score_pct": round(self.score_pct, 3),
            "contextual_risk_level": self.level,
            "contextual_risk_action": self.action,
            "contextual_risk_brake_request_pct": round(
                self.brake_request_pct, 3
            ),
            "contextual_risk_reasons": "|".join(self.reasons),
        }


def _collision_risk(ttc_seconds: object) -> float:
    try:
        ttc = float(ttc_seconds)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(ttc) or ttc < 0.0:
        return 0.0
    if math.isinf(ttc):
        return 0.0
    if ttc <= 1.0:
        return 1.0
    if ttc <= 2.0:
        return 1.0 - 0.20 * (ttc - 1.0)
    if ttc <= 3.0:
        return 0.80 - 0.25 * (ttc - 2.0)
    if ttc <= 5.0:
        return 0.55 - 0.35 * ((ttc - 3.0) / 2.0)
    if ttc <= 8.0:
        return 0.20 - 0.15 * ((ttc - 5.0) / 3.0)
    return 0.0


class ContextualRiskPolicy:
    """Fuse raw TTC urgency with the calibrated five-state DMS output."""

    version = POLICY_VERSION

    def evaluate(self, c1: Any, c2: Any) -> ContextualRiskDecision:
        ttc = float(c1.predicted_ttc_s)
        signals = c2.vss_signals()
        attentive = float(signals.attentive_probability) / 100.0
        distraction = float(signals.distraction_level) / 100.0
        fatigue = float(signals.fatigue_level) / 100.0
        evidence = (attentive, distraction, fatigue)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in evidence):
            raise ValueError("C2 VSS probabilities must be finite percentages in [0, 100]")
        driver_risk = max(1.0 - attentive, distraction, fatigue)
        collision_risk = _collision_risk(ttc)
        score_pct = 100.0 * min(
            1.0, 0.75 * collision_risk + 0.45 * driver_risk
        )

        reasons: list[str] = []
        if math.isfinite(ttc) and ttc < 2.5:
            reasons.append("LOW_TTC")
        state = str(c2.state).lower()
        if state != "alert":
            reasons.append(state.upper())
        if not reasons:
            reasons.append("NORMAL")

        if (math.isfinite(ttc) and ttc <= 1.2) or (
            math.isfinite(ttc) and ttc < 2.0 and driver_risk >= 0.70
        ):
            level = "CRITICAL"
            action = "EMERGENCY_BRAKE_REQUEST"
            brake_request = 70.0
        elif (math.isfinite(ttc) and ttc < 2.5) or score_pct >= 75.0:
            level = "HIGH"
            action = "VISUAL_AUDIO_HAPTIC_WARNING"
            brake_request = 0.0
        elif score_pct >= 45.0:
            level = "CAUTION"
            action = "VISUAL_WARNING"
            brake_request = 0.0
        else:
            level = "SAFE"
            action = "MONITOR"
            brake_request = 0.0
        return ContextualRiskDecision(
            score_pct=round(score_pct, 3),
            level=level,
            action=action,
            brake_request_pct=brake_request,
            reasons=tuple(reasons),
        )


__all__ = [
    "ContextualRiskDecision",
    "ContextualRiskPolicy",
    "POLICY_VERSION",
]
