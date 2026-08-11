package com.fptautomotive.safeloop;

import java.util.Arrays;
import java.util.Collections;

final class DecisionFixtures {
    private DecisionFixtures() {}

    static DecisionSnapshot live(String session, long sequence, long decidedMs, int ttlMs) {
        return DecisionSnapshot.builder()
                .sourceMode(DecisionSnapshot.SourceMode.REPLAY)
                .sessionId(session)
                .sequence(sequence)
                .frameId(sequence + 4L)
                .sourceTimestampMs(sequence * 50L + 200L)
                .decisionTimestampMs(decidedMs)
                .ttlMs(ttlMs)
                .expiresAtMs(decidedMs + ttlMs)
                .validity(true, true, true, true, true, true, true, true)
                .c1(2.25, true, 42.0, false, true, sequence + 4L, 0L)
                .c2("alert", 91.0, 91.0, 6.0, 8.0, Boolean.TRUE, false)
                .c3(72.0, "C", "PREFIX", "hackathon-evaluator-v1-no-tailgating", true)
                .driveQuality(true, 88.0, "B", "PREFIX", false,
                        "safeloop-drive-quality-v1")
                .contextualRisk(48.0, "CAUTION", "VISUAL_WARNING", 0.0,
                        Collections.singletonList("LOW_TTC"), false)
                .health("NOMINAL", true, Collections.emptyList())
                .build();
    }

    static DecisionSnapshot degraded(String session, long decidedMs, int ttlMs) {
        return DecisionSnapshot.builder()
                .sourceMode(DecisionSnapshot.SourceMode.LIVE)
                .sessionId(session)
                .sequence(0L)
                .frameId(4L)
                .sourceTimestampMs(200L)
                .decisionTimestampMs(decidedMs)
                .ttlMs(ttlMs)
                .expiresAtMs(decidedMs + ttlMs)
                .validity(true, true, false, true, false, true, true, false)
                .c1(2.25, true, 42.0, false, true, 4L, 0L)
                .c2("unavailable", Double.NaN, Double.NaN, Double.NaN,
                        Double.NaN, null, false)
                .c3(72.0, "C", "PREFIX", "hackathon-evaluator-v1-no-tailgating", true)
                .driveQuality(true, 88.0, "B", "PREFIX", false,
                        "safeloop-drive-quality-v1")
                .contextualRisk(Double.NaN, "UNAVAILABLE", "MONITOR", 0.0,
                        Collections.singletonList("STALE_OR_INVALID_INPUT"), false)
                .health("DEGRADED", false,
                        Arrays.asList("driver_camera", "c2", "contextual_risk"))
                .build();
    }

    static DecisionSnapshot ancillaryDegraded(String session, long decidedMs, int ttlMs) {
        return DecisionSnapshot.builder()
                .sourceMode(DecisionSnapshot.SourceMode.LIVE)
                .sessionId(session).sequence(0L).frameId(4L).sourceTimestampMs(200L)
                .decisionTimestampMs(decidedMs).ttlMs(ttlMs)
                .expiresAtMs(decidedMs + ttlMs)
                .validity(true, true, true, true, true, true, false, true)
                .c1(1.8, true, 82.0, true, true, 4L, 0L)
                .c2("alert", 91.0, 91.0, 6.0, 8.0, Boolean.TRUE, false)
                .c3(72.0, "C", "PREFIX", "hackathon-evaluator-v1-no-tailgating", true)
                .driveQuality(false, Double.NaN, "N/A", "NO_DATA", false,
                        "safeloop-drive-quality-v1")
                .contextualRisk(80.0, "HIGH", "VISUAL_AUDIO_HAPTIC_WARNING", 0.0,
                        Collections.singletonList("LOW_TTC"), false)
                .health("DEGRADED", true, Collections.singletonList("drive_quality"))
                .build();
    }

    static String validJson() {
        return "{"
                + "\"schema_version\":\"safeloop.decision.v1\","
                + "\"source_mode\":\"replay\","
                + "\"session_id\":\"trip:T01:run-1\","
                + "\"sequence\":0,\"frame_id\":4,"
                + "\"source_timestamp_ms\":200,\"decision_timestamp_ms\":10000,"
                + "\"ttl_ms\":200,\"expires_at_ms\":10200,"
                + "\"validity\":{\"ego\":true,\"front_camera\":true,"
                + "\"driver_camera\":true,\"c1\":true,\"c2\":true,\"c3\":true,"
                + "\"drive_quality\":true,\"contextual_risk\":true},"
                + "\"c1\":{\"ttc_ms\":2250,\"ttc_valid\":true,"
                + "\"collision_probability_pct\":42.0,\"warning\":false,"
                + "\"model_updated\":true,\"model_frame_id\":4,\"age_ms\":0},"
                + "\"c2\":{\"state\":\"alert\",\"confidence_pct\":91.0,"
                + "\"attentive_probability_pct\":91.0,\"distraction_level_pct\":6.0,"
                + "\"fatigue_level_pct\":8.0,\"eyes_on_road\":true,\"warning\":false},"
                + "\"c3\":{\"safe_score_estimate_pct\":72.0,\"grade\":\"C\","
                + "\"scope\":\"PREFIX\",\"formula_version\":\"hackathon-evaluator-v1-no-tailgating\","
                + "\"tailgating_penalty_omitted\":true},"
                + "\"drive_quality\":{\"score_available\":true,\"score_pct\":88.0,"
                + "\"grade\":\"B\",\"scope\":\"PREFIX\",\"window_ready\":false,"
                + "\"formula_version\":\"safeloop-drive-quality-v1\"},"
                + "\"contextual_risk\":{\"score_pct\":48.0,\"level\":\"CAUTION\","
                + "\"action\":\"VISUAL_WARNING\",\"brake_request_pct\":0.0,"
                + "\"reasons\":[\"LOW_TTC\"],\"actuation_authorized\":false},"
                + "\"health\":{\"mode\":\"NOMINAL\",\"decision_valid\":true,"
                + "\"stale_or_invalid_components\":[],"
                + "\"component_age_ms\":{\"c1\":0,\"c2\":0,\"ego\":0},"
                + "\"ttl_ms\":200,\"model_versions\":{\"c1\":\"student-ttc\"}}"
                + "}";
    }
}
