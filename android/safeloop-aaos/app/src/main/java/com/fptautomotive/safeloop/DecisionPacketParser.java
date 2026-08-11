package com.fptautomotive.safeloop;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

/** Strict parser for the Python {@code DecisionEnvelope.to_json_bytes()} contract. */
public final class DecisionPacketParser {
    /** 1500-byte Ethernet MTU minus the 20-byte IPv4 and 8-byte UDP headers. */
    public static final int MAX_PACKET_BYTES = 1_472;

    public DecisionSnapshot parse(String payload) throws PacketFormatException {
        if (payload == null || payload.trim().isEmpty()) {
            throw new PacketFormatException("packet is empty");
        }
        if (payload.getBytes(java.nio.charset.StandardCharsets.UTF_8).length > MAX_PACKET_BYTES) {
            throw new PacketFormatException("packet exceeds " + MAX_PACKET_BYTES + " bytes");
        }
        try {
            JSONObject root = new JSONObject(payload);
            requireExactKeys(root, "root",
                    "schema_version", "source_mode", "session_id", "sequence", "frame_id",
                    "source_timestamp_ms", "decision_timestamp_ms", "ttl_ms", "expires_at_ms",
                    "validity", "c1", "c2", "c3", "drive_quality", "contextual_risk", "health");

            JSONObject validity = object(root, "validity");
            requireExactKeys(validity, "validity", "ego", "front_camera", "driver_camera",
                    "c1", "c2", "c3", "drive_quality", "contextual_risk");
            boolean egoValid = bool(validity, "ego");
            boolean frontValid = bool(validity, "front_camera");
            boolean driverValid = bool(validity, "driver_camera");
            boolean c1Valid = bool(validity, "c1");
            boolean c2Valid = bool(validity, "c2");
            boolean c3Valid = bool(validity, "c3");
            boolean qualityValid = bool(validity, "drive_quality");
            boolean riskValid = bool(validity, "contextual_risk");

            JSONObject c1 = object(root, "c1");
            requireExactKeys(c1, "c1", "ttc_ms", "ttc_valid", "collision_probability_pct",
                    "warning", "model_updated", "model_frame_id", "age_ms");
            Long ttcMs = optionalLong(c1, "ttc_ms");

            JSONObject c2 = object(root, "c2");
            requireExactKeys(c2, "c2", "state", "confidence_pct",
                    "attentive_probability_pct", "distraction_level_pct", "fatigue_level_pct",
                    "eyes_on_road", "warning");

            JSONObject c3 = object(root, "c3");
            requireExactKeys(c3, "c3", "safe_score_estimate_pct", "grade", "scope",
                    "formula_version", "tailgating_penalty_omitted");

            JSONObject quality = object(root, "drive_quality");
            requireExactKeys(quality, "drive_quality", "score_available", "score_pct", "grade",
                    "scope", "window_ready", "formula_version");

            JSONObject risk = object(root, "contextual_risk");
            requireExactKeys(risk, "contextual_risk", "score_pct", "level", "action",
                    "brake_request_pct", "reasons", "actuation_authorized");

            JSONObject health = object(root, "health");
            requireExactKeys(health, "health", "mode", "decision_valid",
                    "stale_or_invalid_components", "component_age_ms", "ttl_ms", "model_versions");
            int ttlMs = integer(root, "ttl_ms", 1, DecisionSnapshot.MAX_TTL_MS);
            if (integer(health, "ttl_ms", 1, DecisionSnapshot.MAX_TTL_MS) != ttlMs) {
                throw new IllegalArgumentException("health.ttl_ms must match ttl_ms");
            }
            validateNonNegativeMap(object(health, "component_age_ms"),
                    "health.component_age_ms", true);
            validateStringMap(object(health, "model_versions"), "health.model_versions");

            List<String> invalidComponents = stringList(
                    health.getJSONArray("stale_or_invalid_components"),
                    "health.stale_or_invalid_components", 16);
            List<String> expectedInvalid = new ArrayList<>();
            if (!egoValid) expectedInvalid.add("ego");
            if (!frontValid) expectedInvalid.add("front_camera");
            if (!driverValid) expectedInvalid.add("driver_camera");
            if (!c1Valid) expectedInvalid.add("c1");
            if (!c2Valid) expectedInvalid.add("c2");
            if (!c3Valid) expectedInvalid.add("c3");
            if (!qualityValid) expectedInvalid.add("drive_quality");
            if (!riskValid) expectedInvalid.add("contextual_risk");
            if (!expectedInvalid.equals(invalidComponents)) {
                throw new IllegalArgumentException(
                        "health stale component list must match validity flags");
            }

            Boolean eyesOnRoad = optionalBoolean(c2, "eyes_on_road");
            List<String> reasons = stringList(
                    risk.getJSONArray("reasons"), "contextual_risk.reasons", 16);
            DecisionSnapshot.Builder builder = DecisionSnapshot.builder()
                    .schemaVersion(text(root, "schema_version"))
                    .sourceMode(DecisionSnapshot.SourceMode.parse(text(root, "source_mode")))
                    .sessionId(text(root, "session_id"))
                    .sequence(nonNegativeLong(root, "sequence"))
                    .frameId(nonNegativeLong(root, "frame_id"))
                    .sourceTimestampMs(nonNegativeLong(root, "source_timestamp_ms"))
                    .decisionTimestampMs(nonNegativeLong(root, "decision_timestamp_ms"))
                    .ttlMs(ttlMs)
                    .expiresAtMs(nonNegativeLong(root, "expires_at_ms"))
                    .validity(egoValid, frontValid, driverValid, c1Valid, c2Valid, c3Valid,
                            qualityValid, riskValid)
                    .c1(ttcMs == null ? Double.POSITIVE_INFINITY : ttcMs / 1000.0,
                            bool(c1, "ttc_valid"), optionalPercent(c1, "collision_probability_pct"),
                            bool(c1, "warning"), bool(c1, "model_updated"),
                            nonNegativeLong(c1, "model_frame_id"), nonNegativeLong(c1, "age_ms"))
                    .c2(text(c2, "state"), optionalPercent(c2, "confidence_pct"),
                            optionalPercent(c2, "attentive_probability_pct"),
                            optionalPercent(c2, "distraction_level_pct"),
                            optionalPercent(c2, "fatigue_level_pct"), eyesOnRoad,
                            bool(c2, "warning"))
                    .c3(optionalPercent(c3, "safe_score_estimate_pct"), text(c3, "grade"),
                            text(c3, "scope"), text(c3, "formula_version"),
                            bool(c3, "tailgating_penalty_omitted"))
                    .driveQuality(bool(quality, "score_available"),
                            optionalPercent(quality, "score_pct"), text(quality, "grade"),
                            text(quality, "scope"), bool(quality, "window_ready"),
                            text(quality, "formula_version"))
                    .contextualRisk(optionalPercent(risk, "score_pct"), text(risk, "level"),
                            text(risk, "action"), percent(risk, "brake_request_pct"), reasons,
                            bool(risk, "actuation_authorized"))
                    .health(text(health, "mode"), bool(health, "decision_valid"), invalidComponents);
            return builder.build();
        } catch (JSONException | IllegalArgumentException error) {
            throw new PacketFormatException("invalid decision packet: " + error.getMessage(), error);
        }
    }

    private static JSONObject object(JSONObject parent, String key) throws JSONException {
        Object value = parent.get(key);
        if (!(value instanceof JSONObject)) {
            throw new JSONException(key + " must be an object");
        }
        return (JSONObject) value;
    }

    private static String text(JSONObject object, String key) throws JSONException {
        Object value = object.get(key);
        if (!(value instanceof String)) {
            throw new JSONException(key + " must be a string");
        }
        return (String) value;
    }

    private static boolean bool(JSONObject object, String key) throws JSONException {
        Object value = object.get(key);
        if (!(value instanceof Boolean)) {
            throw new JSONException(key + " must be boolean");
        }
        return (Boolean) value;
    }

    private static Boolean optionalBoolean(JSONObject object, String key) throws JSONException {
        Object value = object.get(key);
        if (value == JSONObject.NULL) {
            return null;
        }
        if (!(value instanceof Boolean)) {
            throw new JSONException(key + " must be null or boolean");
        }
        return (Boolean) value;
    }

    private static long nonNegativeLong(JSONObject object, String key) throws JSONException {
        long result = wholeNumber(object, key);
        if (result < 0L) {
            throw new JSONException(key + " must be non-negative");
        }
        return result;
    }

    private static Long optionalLong(JSONObject object, String key) throws JSONException {
        if (object.get(key) == JSONObject.NULL) {
            return null;
        }
        return nonNegativeLong(object, key);
    }

    private static int integer(JSONObject object, String key, int minimum, int maximum)
            throws JSONException {
        long value = wholeNumber(object, key);
        if (value < minimum || value > maximum) {
            throw new JSONException(key + " must be in [" + minimum + ", " + maximum + "]");
        }
        return (int) value;
    }

    private static long wholeNumber(JSONObject object, String key) throws JSONException {
        Object raw = object.get(key);
        if (!(raw instanceof Integer) && !(raw instanceof Long)) {
            throw new JSONException(key + " must be an integer");
        }
        return ((Number) raw).longValue();
    }

    private static double percent(JSONObject object, String key) throws JSONException {
        double result = number(object, key);
        if (result < 0.0 || result > 100.0) {
            throw new JSONException(key + " must be in [0, 100]");
        }
        return result;
    }

    private static double optionalPercent(JSONObject object, String key) throws JSONException {
        return object.get(key) == JSONObject.NULL ? Double.NaN : percent(object, key);
    }

    private static double number(JSONObject object, String key) throws JSONException {
        Object raw = object.get(key);
        if (!(raw instanceof Number)) {
            throw new JSONException(key + " must be numeric");
        }
        double result = ((Number) raw).doubleValue();
        if (!Double.isFinite(result)) {
            throw new JSONException(key + " must be finite");
        }
        return result;
    }

    private static List<String> stringList(JSONArray values, String field, int maximum)
            throws JSONException {
        if (values.length() > maximum) {
            throw new JSONException(field + " contains too many items");
        }
        List<String> result = new ArrayList<>(values.length());
        for (int index = 0; index < values.length(); index++) {
            Object value = values.get(index);
            if (!(value instanceof String) || ((String) value).trim().isEmpty()) {
                throw new JSONException(field + " must contain non-empty strings");
            }
            result.add((String) value);
        }
        return result;
    }

    private static void validateNonNegativeMap(JSONObject values, String field, boolean allowEmpty)
            throws JSONException {
        Iterator<String> keys = values.keys();
        int count = 0;
        while (keys.hasNext()) {
            String key = keys.next();
            if (key.trim().isEmpty()) {
                throw new JSONException(field + " keys must not be blank");
            }
            nonNegativeLong(values, key);
            count++;
        }
        if (!allowEmpty && count == 0) {
            throw new JSONException(field + " must not be empty");
        }
    }

    private static void validateStringMap(JSONObject values, String field) throws JSONException {
        Iterator<String> keys = values.keys();
        while (keys.hasNext()) {
            String key = keys.next();
            if (key.trim().isEmpty() || !(values.get(key) instanceof String)
                    || ((String) values.get(key)).trim().isEmpty()) {
                throw new JSONException(field + " must map non-empty strings");
            }
        }
    }

    private static void requireExactKeys(JSONObject object, String field, String... names)
            throws JSONException {
        Set<String> expected = new HashSet<>(Arrays.asList(names));
        Set<String> actual = new HashSet<>();
        Iterator<String> keys = object.keys();
        while (keys.hasNext()) {
            actual.add(keys.next());
        }
        if (!actual.equals(expected)) {
            throw new JSONException(field + " keys do not match decision contract");
        }
    }

    public static final class PacketFormatException extends Exception {
        public PacketFormatException(String message) {
            super(message);
        }

        public PacketFormatException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
