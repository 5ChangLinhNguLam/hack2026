package com.fptautomotive.safeloop;

import org.junit.Test;

import java.util.Collections;

import static org.junit.Assert.assertEquals;

public final class AlertPolicyTest {
    private final AlertPolicy policy = new AlertPolicy();

    @Test
    public void noDataStaleAndDegradedNeverProduceAudio() {
        DecisionStateMachine machine = new DecisionStateMachine();
        assertEquals(AlertPolicy.AudioCue.NONE, policy.evaluate(machine.stateAt(0L)).audioCue);

        machine.accept(DecisionFixtures.degraded("run-a", 10_000L, 200), 1L);
        assertEquals(AlertPolicy.AudioCue.NONE, policy.evaluate(machine.stateAt(1L)).audioCue);
        assertEquals(AlertPolicy.AudioCue.NONE, policy.evaluate(machine.stateAt(201L)).audioCue);
    }

    @Test
    public void emergencyRecommendationProducesCriticalAdvisory() {
        DecisionSnapshot base = DecisionFixtures.live("run-a", 0L, 10_000L, 200);
        DecisionSnapshot critical = DecisionSnapshot.builder()
                .sourceMode(base.sourceMode).sessionId(base.sessionId).sequence(base.sequence)
                .frameId(base.frameId).sourceTimestampMs(base.sourceTimestampMs)
                .decisionTimestampMs(base.decisionTimestampMs).ttlMs(base.ttlMs)
                .expiresAtMs(base.expiresAtMs)
                .validity(true, true, true, true, true, true, true, true)
                .c1(0.9, true, 95.0, true, true, base.c1ModelFrameId, 0L)
                .c2("alert", 91.0, 91.0, 6.0, 8.0, Boolean.TRUE, false)
                .c3(72.0, "C", "PREFIX", base.c3FormulaVersion, true)
                .driveQuality(true, 88.0, "B", "PREFIX", false,
                        base.driveQualityFormulaVersion)
                .contextualRisk(95.0, "CRITICAL", "EMERGENCY_BRAKE_REQUEST", 70.0,
                        Collections.singletonList("LOW_TTC"), false)
                .health("NOMINAL", true, Collections.emptyList())
                .build();
        DecisionStateMachine machine = new DecisionStateMachine();
        machine.accept(critical, 1L);

        AlertPolicy.Decision decision = policy.evaluate(machine.stateAt(1L));
        assertEquals(AlertPolicy.Severity.CRITICAL, decision.severity);
        assertEquals(AlertPolicy.AudioCue.CRITICAL, decision.audioCue);
    }

    @Test
    public void ancillaryDegradationDoesNotSuppressValidCollisionWarning() {
        DecisionStateMachine machine = new DecisionStateMachine();
        machine.accept(DecisionFixtures.ancillaryDegraded("run-a", 10_000L, 200), 1L);

        DashboardState state = machine.stateAt(1L);
        AlertPolicy.Decision decision = policy.evaluate(state);
        assertEquals(DashboardState.Health.DEGRADED, state.health);
        assertEquals(AlertPolicy.Severity.HIGH, decision.severity);
        assertEquals(AlertPolicy.AudioCue.HIGH, decision.audioCue);
    }
}
