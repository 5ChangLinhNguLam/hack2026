package com.fptautomotive.safeloop;

import java.util.Locale;

/** Visual/audio policy only.  This class never emits a vehicle command. */
public final class AlertPolicy {
    public enum Severity {
        NONE,
        SAFE,
        CAUTION,
        HIGH,
        CRITICAL
    }

    public enum AudioCue {
        NONE,
        HIGH,
        CRITICAL
    }

    public static final class Decision {
        public final Severity severity;
        public final AudioCue audioCue;
        public final String label;
        public final String detail;

        Decision(Severity severity, AudioCue audioCue, String label, String detail) {
            this.severity = severity;
            this.audioCue = audioCue;
            this.label = label;
            this.detail = detail;
        }
    }

    public Decision evaluate(DashboardState state) {
        if (state == null || state.health == DashboardState.Health.NO_DATA) {
            return new Decision(Severity.NONE, AudioCue.NONE, "NO DATA", "Waiting for decision bridge");
        }
        if (state.health == DashboardState.Health.STALE) {
            return new Decision(Severity.NONE, AudioCue.NONE, "STALE", "Alerts suppressed: input expired");
        }
        DecisionSnapshot packet = state.snapshot;
        if (packet == null || !packet.contextualRiskValid || !packet.decisionValid) {
            String detail = state.issue.isEmpty()
                    ? "Alerts suppressed: contextual decision unavailable"
                    : state.issue;
            return new Decision(Severity.NONE, AudioCue.NONE, "DEGRADED", detail);
        }

        String action = packet.action.toUpperCase(Locale.ROOT);
        String declaredLevel = packet.riskLevel.toUpperCase(Locale.ROOT);
        boolean finiteTtc = packet.c1Valid && packet.ttcValid
                && Double.isFinite(packet.ttcSeconds);
        if (declaredLevel.equals("CRITICAL")
                || (finiteTtc && packet.ttcSeconds <= 1.2)
                || packet.contextRiskPct >= 90.0) {
            return new Decision(
                    Severity.CRITICAL,
                    AudioCue.CRITICAL,
                    "COLLISION RISK",
                    "Critical driver warning; vehicle control remains outside HMI");
        }
        if (action.equals(DecisionSnapshot.WARNING_ONLY_ACTION)
                || declaredLevel.equals("HIGH")
                || (finiteTtc && packet.ttcSeconds < 2.5)
                || packet.contextRiskPct >= 70.0) {
            return new Decision(
                    Severity.HIGH, AudioCue.HIGH, "TAKE CONTROL", humanize(packet.action));
        }
        if (action.equals("VISUAL_WARNING")
                || declaredLevel.equals("CAUTION")
                || packet.contextRiskPct >= 40.0) {
            return new Decision(
                    Severity.CAUTION, AudioCue.NONE, "CAUTION", humanize(packet.action));
        }
        return new Decision(Severity.SAFE, AudioCue.NONE, "SAFE", "Monitoring");
    }

    private static String humanize(String value) {
        return value.replace('_', ' ');
    }
}
