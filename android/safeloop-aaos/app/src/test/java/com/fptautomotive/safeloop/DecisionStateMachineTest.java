package com.fptautomotive.safeloop;

import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public final class DecisionStateMachineTest {
    @Test
    public void freshEnvelopeExpiresExactlyAtItsTtl() {
        DecisionStateMachine machine = new DecisionStateMachine();
        assertEquals(DashboardState.Health.NO_DATA, machine.stateAt(100L).health);

        assertTrue(machine.accept(
                DecisionFixtures.live("run-a", 0L, 10_000L, 200), 5_000L));
        assertEquals(DashboardState.Health.LIVE, machine.stateAt(5_199L).health);
        assertEquals(DashboardState.Health.STALE, machine.stateAt(5_200L).health);
    }

    @Test
    public void ignoresClockDomainAndLateJoinsAtAnySequence() {
        DecisionStateMachine machine = new DecisionStateMachine();
        assertTrue(machine.accept(
                DecisionFixtures.live("clock-independent", 0L, 99_000L, 200), 1L));
        assertTrue(machine.accept(
                DecisionFixtures.live("bad-start", 3L, 10_000L, 200), 3L));

        assertTrue(machine.accept(
                DecisionFixtures.live("run-a", 7L, 10_000L, 200), 4L));
        assertFalse(machine.accept(
                DecisionFixtures.live("run-a", 7L, 10_000L, 200), 5L));
        assertTrue(machine.accept(
                DecisionFixtures.live("run-a", 10L, 10_010L, 200), 6L));
        DashboardState state = machine.stateAt(6L);
        assertEquals(1L, state.rejectedPackets);
        assertEquals(2L, state.droppedPackets);
    }

    @Test
    public void countsSequenceGapsAndRejectsRetiredSession() {
        DecisionStateMachine machine = new DecisionStateMachine();
        assertTrue(machine.accept(
                DecisionFixtures.live("run-a", 0L, 10_000L, 200), 100L));
        assertTrue(machine.accept(
                DecisionFixtures.live("run-a", 3L, 10_010L, 200), 110L));
        assertEquals(2L, machine.stateAt(110L).droppedPackets);

        assertTrue(machine.accept(
                DecisionFixtures.live("run-b", 0L, 10_020L, 200), 120L));
        assertEquals(1L, machine.stateAt(120L).sessionRestarts);
        assertFalse(machine.accept(
                DecisionFixtures.live("run-a", 0L, 10_030L, 200), 130L));
    }

    @Test
    public void contractDegradedStateIsSilentAndNeverPromotedToLive() {
        DecisionStateMachine machine = new DecisionStateMachine();
        assertTrue(machine.accept(
                DecisionFixtures.degraded("run-a", 10_000L, 200), 100L));
        DashboardState state = machine.stateAt(100L);
        assertEquals(DashboardState.Health.DEGRADED, state.health);
        assertTrue(state.issue.contains("driver_camera"));
    }

    @Test
    public void boundsRememberedSessionsInsteadOfGrowingForever() {
        DecisionStateMachine machine = new DecisionStateMachine();
        for (int index = 0; index < 70; index++) {
            assertTrue(machine.accept(
                    DecisionFixtures.live("run-" + index, 5L, 10_000L + index, 200),
                    100L + index));
        }
        // run-68 is still inside the recent retired-session replay window.
        assertFalse(machine.accept(
                DecisionFixtures.live("run-68", 0L, 11_000L, 200), 170L));
        // run-0 was evicted from the bounded LRU, so streaming does not stop
        // permanently after 64 distinct sessions.
        assertTrue(machine.accept(
                DecisionFixtures.live("run-0", 9L, 11_010L, 200), 171L));
    }

    @Test
    public void reusedRetiredSessionRestartsOnlyAfterCurrentStreamIsStale() {
        DecisionStateMachine machine = new DecisionStateMachine();
        assertTrue(machine.accept(
                DecisionFixtures.live("run-a", 0L, 10_000L, 200), 100L));
        assertTrue(machine.accept(
                DecisionFixtures.live("run-b", 0L, 10_010L, 200), 110L));

        assertFalse(machine.accept(
                DecisionFixtures.live("run-a", 0L, 10_020L, 200), 309L));
        assertTrue(machine.accept(
                DecisionFixtures.live("run-a", 0L, 10_030L, 200), 310L));
        DashboardState state = machine.stateAt(310L);
        assertEquals(DashboardState.Health.LIVE, state.health);
        assertEquals(2L, state.sessionRestarts);
    }

    @Test
    public void reusedActiveSessionSequenceZeroRestartsOnlyAfterStale() {
        DecisionStateMachine machine = new DecisionStateMachine();
        assertTrue(machine.accept(
                DecisionFixtures.live("run-a", 7L, 10_000L, 200), 100L));
        assertFalse(machine.accept(
                DecisionFixtures.live("run-a", 0L, 10_010L, 200), 299L));
        assertTrue(machine.accept(
                DecisionFixtures.live("run-a", 0L, 10_020L, 200), 300L));
        assertEquals(1L, machine.stateAt(300L).sessionRestarts);
    }
}
