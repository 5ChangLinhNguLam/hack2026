package com.fptautomotive.safeloop;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

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
    public void legacyEmergencyInputIsNormalizedBeforeCriticalAlert() {
        DecisionSnapshot critical = DecisionFixtures.legacyEmergency(
                "run-a", 10_000L, 200);
        assertEquals(DecisionSnapshot.WARNING_ONLY_ACTION, critical.action);
        assertEquals(0.0, critical.brakeRequestPct, 0.0);
        assertFalse(critical.actuationAuthorized);
        DecisionStateMachine machine = new DecisionStateMachine();
        machine.accept(critical, 1L);

        AlertPolicy.Decision decision = policy.evaluate(machine.stateAt(1L));
        assertEquals(AlertPolicy.Severity.CRITICAL, decision.severity);
        assertEquals(AlertPolicy.AudioCue.CRITICAL, decision.audioCue);
        assertFalse(decision.detail.toLowerCase(java.util.Locale.ROOT).contains("brake"));
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
