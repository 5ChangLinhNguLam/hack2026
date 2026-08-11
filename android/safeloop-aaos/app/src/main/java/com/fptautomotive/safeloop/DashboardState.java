package com.fptautomotive.safeloop;

/** Immutable receiver-health and decision state consumed by the Canvas view. */
public final class DashboardState {
    public enum Health {
        NO_DATA,
        LIVE,
        DEGRADED,
        STALE
    }

    public final Health health;
    public final DecisionSnapshot snapshot;
    public final long packetAgeMs;
    public final long droppedPackets;
    public final long rejectedPackets;
    public final long parseErrors;
    public final long sessionRestarts;
    public final String issue;

    DashboardState(
            Health health,
            DecisionSnapshot snapshot,
            long packetAgeMs,
            long droppedPackets,
            long rejectedPackets,
            long parseErrors,
            long sessionRestarts,
            String issue) {
        this.health = health;
        this.snapshot = snapshot;
        this.packetAgeMs = packetAgeMs;
        this.droppedPackets = droppedPackets;
        this.rejectedPackets = rejectedPackets;
        this.parseErrors = parseErrors;
        this.sessionRestarts = sessionRestarts;
        this.issue = issue == null ? "" : issue;
    }

    public boolean hasData() {
        return snapshot != null;
    }
}
