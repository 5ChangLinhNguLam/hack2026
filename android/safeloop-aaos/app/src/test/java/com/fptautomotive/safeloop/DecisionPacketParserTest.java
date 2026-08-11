package com.fptautomotive.safeloop;

import org.json.JSONObject;
import org.json.JSONArray;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public final class DecisionPacketParserTest {
    private final DecisionPacketParser parser = new DecisionPacketParser();

    @Test
    public void parsesExactPythonEnvelope() throws Exception {
        DecisionSnapshot result = parser.parse(DecisionFixtures.validJson());

        assertEquals(DecisionSnapshot.SourceMode.REPLAY, result.sourceMode);
        assertEquals("trip:T01:run-1", result.sessionId);
        assertEquals(2.25, result.ttcSeconds, 0.0001);
        assertEquals(72.0, result.c3SafeScorePct, 0.0001);
        assertEquals(88.0, result.driveQualityPct, 0.0001);
        assertTrue(result.driveQualityAvailable);
        assertFalse(result.actuationAuthorized);
    }

    @Test
    public void rejectsPacketAboveNonFragmentingUdpLimit() throws Exception {
        assertEquals(1_472, DecisionPacketParser.MAX_PACKET_BYTES);
        JSONObject oversized = new JSONObject(DecisionFixtures.validJson());
        oversized.getJSONObject("health")
                .getJSONObject("model_versions")
                .put("c1", "x".repeat(DecisionPacketParser.MAX_PACKET_BYTES));

        assertTrue(
                oversized.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8).length
                        > DecisionPacketParser.MAX_PACKET_BYTES);
        assertRejected(oversized.toString(), "exceeds 1472 bytes");
    }

    @Test
    public void parsesNullTtcAsUnavailableNotInfinityJson() throws Exception {
        JSONObject json = new JSONObject(DecisionFixtures.validJson());
        json.getJSONObject("c1").put("ttc_ms", JSONObject.NULL).put("ttc_valid", false);

        DecisionSnapshot result = parser.parse(json.toString());
        assertFalse(result.ttcValid);
        assertEquals(Double.POSITIVE_INFINITY, result.ttcSeconds, 0.0);
    }

    @Test
    public void acceptsNonNegativeTtcBeyondAndroidDisplayRange() throws Exception {
        JSONObject json = new JSONObject(DecisionFixtures.validJson());
        json.getJSONObject("c1").put("ttc_ms", 301_000L);

        DecisionSnapshot result = parser.parse(json.toString());
        assertTrue(result.ttcValid);
        assertEquals(301.0, result.ttcSeconds, 0.0);
    }

    @Test
    public void rejectsMaliciousInvalidC1LowTtcBeforeItCanAlert() throws Exception {
        JSONObject json = new JSONObject(DecisionFixtures.validJson());
        invalidate(json, "c1");
        json.getJSONObject("c1")
                .put("ttc_ms", 500L)
                .put("ttc_valid", true)
                .put("collision_probability_pct", JSONObject.NULL)
                .put("warning", true);
        json.getJSONObject("contextual_risk")
                .put("score_pct", 5.0)
                .put("level", "SAFE")
                .put("action", "MONITOR")
                .put("reasons", new JSONArray());

        assertRejected(json.toString(), "invalid C1 must suppress");
    }

    @Test
    public void rejectsInvalidC2WarningOrPartialMetrics() throws Exception {
        JSONObject json = new JSONObject(DecisionFixtures.validJson());
        invalidate(json, "c2");
        json.getJSONObject("c2")
                .put("state", "unavailable")
                .put("confidence_pct", JSONObject.NULL)
                .put("attentive_probability_pct", JSONObject.NULL)
                .put("distraction_level_pct", JSONObject.NULL)
                .put("fatigue_level_pct", JSONObject.NULL)
                .put("eyes_on_road", JSONObject.NULL)
                .put("warning", true);

        assertRejected(json.toString(), "unavailable state");

        json.getJSONObject("c2")
                .put("warning", false)
                .put("confidence_pct", 12.0);
        assertRejected(json.toString(), "unavailable state");
    }

    @Test
    public void rejectsC3GradeThatContradictsAvailability() throws Exception {
        JSONObject invalid = new JSONObject(DecisionFixtures.validJson());
        invalidate(invalid, "c3");
        invalid.getJSONObject("c3").put("safe_score_estimate_pct", JSONObject.NULL);
        assertRejected(invalid.toString(), "C3 grade availability");

        JSONObject valid = new JSONObject(DecisionFixtures.validJson());
        valid.getJSONObject("c3").put("grade", "N/A");
        assertRejected(valid.toString(), "C3 grade availability");
    }

    @Test
    public void rejectsInvalidDriveQualityUnlessItIsExplicitNoData() throws Exception {
        JSONObject json = new JSONObject(DecisionFixtures.validJson());
        invalidate(json, "drive_quality");

        assertRejected(json.toString(), "explicit NO_DATA");
    }

    @Test
    public void rejectsContextualRiskLevelThatContradictsAvailability() throws Exception {
        JSONObject invalid = new JSONObject(DecisionFixtures.validJson());
        invalidate(invalid, "contextual_risk");
        invalid.getJSONObject("health").put("decision_valid", false);
        invalid.getJSONObject("contextual_risk")
                .put("score_pct", JSONObject.NULL)
                .put("level", "SAFE")
                .put("action", "MONITOR")
                .put("brake_request_pct", 0.0);
        assertRejected(invalid.toString(), "UNAVAILABLE/MONITOR");

        JSONObject valid = new JSONObject(DecisionFixtures.validJson());
        valid.getJSONObject("contextual_risk").put("level", "UNAVAILABLE");
        assertRejected(valid.toString(), "valid contextual risk");
    }

    @Test
    public void rejectsUnknownKeysAndAuthorizedActuation() throws Exception {
        JSONObject unknown = new JSONObject(DecisionFixtures.validJson()).put("unknown", true);
        assertRejected(unknown.toString(), "keys");

        JSONObject authorized = new JSONObject(DecisionFixtures.validJson());
        authorized.getJSONObject("contextual_risk").put("actuation_authorized", true);
        assertRejected(authorized.toString(), "authorize vehicle actuation");
    }

    @Test
    public void rejectsValidityHealthMismatchAndNonIntegerSequence() throws Exception {
        JSONObject mismatch = new JSONObject(DecisionFixtures.validJson());
        mismatch.getJSONObject("validity").put("c2", false);
        assertRejected(mismatch.toString(), "stale component list");

        JSONObject fractional = new JSONObject(DecisionFixtures.validJson()).put("sequence", 1.5);
        assertRejected(fractional.toString(), "integer");

        String integralFloat = DecisionFixtures.validJson()
                .replace("\"sequence\":0", "\"sequence\":1.0");
        assertRejected(integralFloat, "integer");
    }

    private void assertRejected(String payload, String messageFragment) throws Exception {
        try {
            parser.parse(payload);
            fail("packet should be rejected");
        } catch (DecisionPacketParser.PacketFormatException error) {
            assertTrue(error.getMessage(), error.getMessage().contains(messageFragment));
        }
    }

    private static void invalidate(JSONObject envelope, String component) throws Exception {
        envelope.getJSONObject("validity").put(component, false);
        envelope.getJSONObject("health")
                .put("mode", "DEGRADED")
                .put("stale_or_invalid_components", new JSONArray().put(component));
    }
}
