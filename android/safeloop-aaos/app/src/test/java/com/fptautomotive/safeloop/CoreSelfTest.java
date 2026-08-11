package com.fptautomotive.safeloop;

/** Dependency-free smoke test runnable with only the JDK compiler module. */
public final class CoreSelfTest {
    private CoreSelfTest() {}

    public static void main(String[] args) {
        DecisionStateMachine machine = new DecisionStateMachine();
        require(machine.stateAt(0L).health == DashboardState.Health.NO_DATA, "initial no data");
        require(machine.accept(
                DecisionFixtures.live("run-a", 0L, 10_000L, 200), 1_000L),
                "fresh packet accepted");
        require(machine.stateAt(1_199L).health == DashboardState.Health.LIVE,
                "live before exact TTL");
        require(machine.stateAt(1_200L).health == DashboardState.Health.STALE,
                "stale at exact TTL");
        require(new AlertPolicy().evaluate(machine.stateAt(1_200L)).audioCue
                        == AlertPolicy.AudioCue.NONE,
                "stale state is silent");

        DecisionStateMachine sessions = new DecisionStateMachine();
        require(sessions.accept(
                DecisionFixtures.live("one", 0L, 20_000L, 200), 2_000L),
                "first session accepted");
        require(sessions.accept(
                DecisionFixtures.live("two", 0L, 20_010L, 200), 2_010L),
                "second session accepted");
        require(!sessions.accept(
                DecisionFixtures.live("one", 0L, 20_020L, 200), 2_020L),
                "retired session rejected");
        System.out.println("CoreSelfTest PASS");
    }

    private static void require(boolean condition, String description) {
        if (!condition) {
            throw new AssertionError(description);
        }
    }
}
