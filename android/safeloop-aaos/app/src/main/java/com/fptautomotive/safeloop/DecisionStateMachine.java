package com.fptautomotive.safeloop;

import java.util.LinkedHashMap;

/**
 * Causal receiver guard for the SafeLoop sequence/session contract.
 *
 * <p>Producer timestamps remain evidence fields. Freshness starts when the
 * datagram reaches AAOS and advances only on Android's monotonic clock, so the
 * producer and AAOS VM do not need synchronized wall clocks. The first packet
 * seen for an unknown session establishes its sequence baseline. Replay
 * protection covers the 64 most recently retired sessions; after the active
 * stream is stale, sequence zero may explicitly restart a reused identifier.</p>
 */
public final class DecisionStateMachine {
    private static final int MAX_RETIRED_SESSIONS = 64;
    private DecisionSnapshot latest;
    private long receivedAtElapsedMs = -1L;
    private long expiresAtElapsedMs = -1L;
    private String sessionId = "";
    private long lastSequence = -1L;
    private long droppedPackets;
    private long rejectedPackets;
    private long parseErrors;
    private long sessionRestarts;
    private String issue = "";
    private final LinkedHashMap<String, Boolean> retiredSessions =
            new LinkedHashMap<>(MAX_RETIRED_SESSIONS, 0.75f, true);

    public DecisionStateMachine() {}

    public synchronized DashboardState reset() {
        latest = null;
        receivedAtElapsedMs = -1L;
        expiresAtElapsedMs = -1L;
        sessionId = "";
        lastSequence = -1L;
        droppedPackets = 0L;
        rejectedPackets = 0L;
        parseErrors = 0L;
        sessionRestarts = 0L;
        issue = "";
        retiredSessions.clear();
        return stateAt(0L);
    }

    /** Returns false when freshness, session, or ordering checks reject a packet. */
    public synchronized boolean accept(DecisionSnapshot packet, long receivedElapsedMs) {
        if (packet == null) {
            throw new IllegalArgumentException("packet must not be null");
        }
        requireNonNegative(receivedElapsedMs, "elapsed time");

        boolean newSession = latest == null || !sessionId.equals(packet.sessionId);
        boolean staleRestart = packet.sequence == 0L
                && latest != null
                && receivedElapsedMs >= expiresAtElapsedMs;
        if (newSession && isRetiredSession(packet.sessionId)) {
            if (!staleRestart) {
                return reject("PREVIOUSLY_RETIRED_SESSION");
            }
            retiredSessions.remove(packet.sessionId);
        }
        if (!newSession && packet.sequence <= lastSequence) {
            if (!staleRestart) {
                return reject("DUPLICATE_OR_OUT_OF_ORDER_SEQUENCE");
            }
            sessionRestarts++;
        }
        if (newSession) {
            if (latest != null) {
                retireActiveSession();
                sessionRestarts++;
            }
            sessionId = packet.sessionId;
        } else if (!staleRestart && packet.sequence > lastSequence + 1L) {
            droppedPackets = safeAdd(
                    droppedPackets, packet.sequence - lastSequence - 1L);
        }

        latest = packet;
        lastSequence = packet.sequence;
        receivedAtElapsedMs = receivedElapsedMs;
        expiresAtElapsedMs = safeAdd(receivedElapsedMs, packet.ttlMs);
        issue = packet.allInputsValid() && "NOMINAL".equals(packet.healthMode)
                ? ""
                : joinInvalid(packet);
        return true;
    }

    private boolean isRetiredSession(String candidate) {
        // LinkedHashMap is access ordered: a rejected replay remains inside
        // the finite recent-session protection window.
        return retiredSessions.get(candidate) != null;
    }

    private void retireActiveSession() {
        if (sessionId.isEmpty()) {
            return;
        }
        retiredSessions.put(sessionId, Boolean.TRUE);
        while (retiredSessions.size() > MAX_RETIRED_SESSIONS) {
            String oldest = retiredSessions.keySet().iterator().next();
            retiredSessions.remove(oldest);
        }
    }

    public synchronized void recordParseError(String message) {
        parseErrors++;
        issue = safeIssue(message, "MALFORMED_PACKET");
    }

    public synchronized void recordTransportError(String message) {
        issue = safeIssue(message, "TRANSPORT_ERROR");
    }

    public synchronized DashboardState stateAt(long nowElapsedMs) {
        requireNonNegative(nowElapsedMs, "elapsed time");
        if (latest == null) {
            return state(DashboardState.Health.NO_DATA, null, -1L);
        }
        long age = Math.max(0L, nowElapsedMs - receivedAtElapsedMs);
        if (nowElapsedMs >= expiresAtElapsedMs) {
            return state(DashboardState.Health.STALE, latest, age);
        }
        boolean degraded = !latest.decisionValid
                || !"NOMINAL".equals(latest.healthMode)
                || !latest.allInputsValid();
        return state(degraded ? DashboardState.Health.DEGRADED : DashboardState.Health.LIVE,
                latest, age);
    }

    private DashboardState state(
            DashboardState.Health health, DecisionSnapshot snapshot, long age) {
        return new DashboardState(health, snapshot, age, droppedPackets, rejectedPackets,
                parseErrors, sessionRestarts, issue);
    }

    private boolean reject(String reason) {
        rejectedPackets++;
        issue = reason;
        return false;
    }

    private static String joinInvalid(DecisionSnapshot packet) {
        if (packet.staleOrInvalidComponents.isEmpty()) {
            return "DECISION_DEGRADED";
        }
        return "Invalid: " + String.join(", ", packet.staleOrInvalidComponents);
    }

    private static void requireNonNegative(long value, String field) {
        if (value < 0L) {
            throw new IllegalArgumentException(field + " must be non-negative");
        }
    }

    private static long safeAdd(long left, long right) {
        return left > Long.MAX_VALUE - right ? Long.MAX_VALUE : left + right;
    }

    private static String safeIssue(String value, String fallback) {
        if (value == null || value.trim().isEmpty()) {
            return fallback;
        }
        String result = value.trim().replace('\n', ' ').replace('\r', ' ');
        return result.length() <= 120 ? result : result.substring(0, 120);
    }
}
