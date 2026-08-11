package com.fptautomotive.safeloop;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Objects;

/** Immutable subset of the validated SafeLoop product decision envelope. */
public final class DecisionSnapshot {
    public static final String SCHEMA = "safeloop.decision.v1";
    public static final int MAX_TTL_MS = 1_000;

    public enum SourceMode {
        LIVE,
        REPLAY,
        SIMULATION;

        public static SourceMode parse(String value) {
            try {
                return valueOf(Objects.requireNonNull(value).toUpperCase(Locale.ROOT));
            } catch (IllegalArgumentException | NullPointerException error) {
                throw new IllegalArgumentException("unsupported source_mode: " + value, error);
            }
        }
    }

    public final String schemaVersion;
    public final SourceMode sourceMode;
    public final String sessionId;
    public final long sequence;
    public final long frameId;
    public final long sourceTimestampMs;
    public final long decisionTimestampMs;
    public final int ttlMs;
    public final long expiresAtMs;

    public final boolean egoValid;
    public final boolean frontCameraValid;
    public final boolean driverCameraValid;
    public final boolean c1Valid;
    public final boolean c2Valid;
    public final boolean c3Valid;
    public final boolean driveQualityValid;
    public final boolean contextualRiskValid;

    public final double ttcSeconds;
    public final boolean ttcValid;
    public final double collisionProbabilityPct;
    public final boolean collisionWarning;
    public final boolean c1ModelUpdated;
    public final long c1ModelFrameId;
    public final long c1AgeMs;

    public final String driverState;
    public final double driverConfidencePct;
    public final double attentionPct;
    public final double distractionPct;
    public final double fatiguePct;
    public final Boolean eyesOnRoad;
    public final boolean driverWarning;

    public final double c3SafeScorePct;
    public final String c3Grade;
    public final String c3Scope;
    public final String c3FormulaVersion;
    public final boolean c3TailgatingPenaltyOmitted;

    public final boolean driveQualityAvailable;
    public final double driveQualityPct;
    public final String driveQualityGrade;
    public final String driveQualityScope;
    public final boolean driveQualityWindowReady;
    public final String driveQualityFormulaVersion;

    public final double contextRiskPct;
    public final String riskLevel;
    public final String action;
    public final double brakeRequestPct;
    public final List<String> riskReasons;
    public final boolean actuationAuthorized;

    public final String healthMode;
    public final boolean decisionValid;
    public final List<String> staleOrInvalidComponents;

    private DecisionSnapshot(Builder builder) {
        schemaVersion = requireText(builder.schemaVersion, "schema_version", 64);
        if (!SCHEMA.equals(schemaVersion)) {
            throw new IllegalArgumentException("unsupported schema_version: " + schemaVersion);
        }
        sourceMode = Objects.requireNonNull(builder.sourceMode, "source_mode");
        sessionId = requireText(builder.sessionId, "session_id", 128);
        sequence = requireNonNegative(builder.sequence, "sequence");
        frameId = requireNonNegative(builder.frameId, "frame_id");
        sourceTimestampMs = requireNonNegative(builder.sourceTimestampMs, "source_timestamp_ms");
        decisionTimestampMs = requireNonNegative(builder.decisionTimestampMs, "decision_timestamp_ms");
        if (builder.ttlMs < 1 || builder.ttlMs > MAX_TTL_MS) {
            throw new IllegalArgumentException("ttl_ms must be in [1, 1000]");
        }
        ttlMs = builder.ttlMs;
        expiresAtMs = requireNonNegative(builder.expiresAtMs, "expires_at_ms");
        if (expiresAtMs != decisionTimestampMs + ttlMs) {
            throw new IllegalArgumentException("expires_at_ms must equal decision_timestamp_ms + ttl_ms");
        }

        egoValid = builder.egoValid;
        frontCameraValid = builder.frontCameraValid;
        driverCameraValid = builder.driverCameraValid;
        c1Valid = builder.c1Valid;
        c2Valid = builder.c2Valid;
        c3Valid = builder.c3Valid;
        driveQualityValid = builder.driveQualityValid;
        contextualRiskValid = builder.contextualRiskValid;

        ttcSeconds = builder.ttcSeconds;
        ttcValid = builder.ttcValid;
        if (ttcValid != Double.isFinite(ttcSeconds)) {
            throw new IllegalArgumentException("c1.ttc_valid must match c1.ttc_ms availability");
        }
        if (ttcValid && ttcSeconds < 0.0) {
            throw new IllegalArgumentException("c1.ttc_ms must be non-negative");
        }
        collisionProbabilityPct = optionalPercent(builder.collisionProbabilityPct,
                "c1.collision_probability_pct");
        collisionWarning = builder.collisionWarning;
        c1ModelUpdated = builder.c1ModelUpdated;
        c1ModelFrameId = requireNonNegative(builder.c1ModelFrameId, "c1.model_frame_id");
        c1AgeMs = requireNonNegative(builder.c1AgeMs, "c1.age_ms");
        if (c1ModelFrameId > frameId) {
            throw new IllegalArgumentException("c1.model_frame_id cannot exceed frame_id");
        }
        if (c1Valid != Double.isFinite(collisionProbabilityPct)) {
            throw new IllegalArgumentException(
                    "validity.c1 must match collision probability availability");
        }
        if (!c1Valid && (ttcValid || collisionWarning)) {
            throw new IllegalArgumentException(
                    "invalid C1 must suppress TTC validity and collision warning");
        }

        driverState = requireOneOf(builder.driverState, "c2.state",
                "alert", "drowsy", "microsleep", "yawning", "distracted", "unavailable");
        driverConfidencePct = optionalPercent(builder.driverConfidencePct, "c2.confidence_pct");
        attentionPct = optionalPercent(builder.attentionPct, "c2.attentive_probability_pct");
        distractionPct = optionalPercent(builder.distractionPct, "c2.distraction_level_pct");
        fatiguePct = optionalPercent(builder.fatiguePct, "c2.fatigue_level_pct");
        eyesOnRoad = builder.eyesOnRoad;
        driverWarning = builder.driverWarning;
        boolean allDriverMetricsAvailable = Double.isFinite(driverConfidencePct)
                && Double.isFinite(attentionPct)
                && Double.isFinite(distractionPct)
                && Double.isFinite(fatiguePct)
                && eyesOnRoad != null;
        if (c2Valid) {
            if (!allDriverMetricsAvailable || "unavailable".equals(driverState)) {
                throw new IllegalArgumentException(
                        "validity.c2 must match C2 metric availability and state");
            }
        } else if (!"unavailable".equals(driverState)
                || Double.isFinite(driverConfidencePct)
                || Double.isFinite(attentionPct)
                || Double.isFinite(distractionPct)
                || Double.isFinite(fatiguePct)
                || eyesOnRoad != null
                || driverWarning) {
            throw new IllegalArgumentException(
                    "invalid C2 must expose only the unavailable state");
        }

        c3SafeScorePct = optionalPercent(builder.c3SafeScorePct, "c3.safe_score_estimate_pct");
        c3Grade = requireOneOf(builder.c3Grade, "c3.grade", "A", "B", "C", "D", "E", "N/A");
        c3Scope = requireOneOf(builder.c3Scope, "c3.scope", "PREFIX", "FULL_TRIP");
        c3FormulaVersion = requireText(builder.c3FormulaVersion, "c3.formula_version", 128);
        c3TailgatingPenaltyOmitted = builder.c3TailgatingPenaltyOmitted;
        if (c3Valid == "N/A".equals(c3Grade)) {
            throw new IllegalArgumentException(
                    "C3 grade availability must match validity.c3");
        }

        driveQualityAvailable = builder.driveQualityAvailable;
        driveQualityPct = optionalPercent(builder.driveQualityPct, "drive_quality.score_pct");
        if (driveQualityAvailable != Double.isFinite(driveQualityPct)) {
            throw new IllegalArgumentException(
                    "drive_quality.score_available must match score_pct availability");
        }
        driveQualityGrade = requireOneOf(builder.driveQualityGrade, "drive_quality.grade",
                "A", "B", "C", "D", "E", "N/A");
        driveQualityScope = requireOneOf(builder.driveQualityScope, "drive_quality.scope",
                "NO_DATA", "PREFIX", "FULL_TRIP", "ROLLING_60S");
        driveQualityWindowReady = builder.driveQualityWindowReady;
        driveQualityFormulaVersion = requireText(
                builder.driveQualityFormulaVersion, "drive_quality.formula_version", 128);
        if (!driveQualityValid && (driveQualityAvailable
                || !"N/A".equals(driveQualityGrade)
                || !"NO_DATA".equals(driveQualityScope)
                || driveQualityWindowReady)) {
            throw new IllegalArgumentException(
                    "invalid drive quality must expose an explicit NO_DATA state");
        }

        contextRiskPct = optionalPercent(builder.contextRiskPct, "contextual_risk.score_pct");
        riskLevel = requireOneOf(builder.riskLevel, "contextual_risk.level",
                "SAFE", "CAUTION", "HIGH", "CRITICAL", "UNAVAILABLE");
        action = requireOneOf(builder.action, "contextual_risk.action",
                "MONITOR", "VISUAL_WARNING", "VISUAL_AUDIO_HAPTIC_WARNING",
                "EMERGENCY_BRAKE_REQUEST");
        brakeRequestPct = requirePercent(builder.brakeRequestPct,
                "contextual_risk.brake_request_pct");
        riskReasons = immutableShortList(builder.riskReasons, "contextual_risk.reasons", 16);
        actuationAuthorized = builder.actuationAuthorized;
        if (actuationAuthorized) {
            throw new IllegalArgumentException("Android envelope must never authorize vehicle actuation");
        }
        if (!contextualRiskValid && (!"UNAVAILABLE".equals(riskLevel)
                || !"MONITOR".equals(action) || brakeRequestPct != 0.0)) {
            throw new IllegalArgumentException(
                    "invalid contextual risk must be UNAVAILABLE/MONITOR with zero brake");
        }
        if (contextualRiskValid && "UNAVAILABLE".equals(riskLevel)) {
            throw new IllegalArgumentException(
                    "valid contextual risk cannot have UNAVAILABLE level");
        }
        if ("EMERGENCY_BRAKE_REQUEST".equals(action)
                && !(egoValid && frontCameraValid && c1Valid)) {
            throw new IllegalArgumentException("emergency recommendation requires fresh C1 inputs");
        }

        healthMode = requireOneOf(builder.healthMode, "health.mode",
                "INITIALIZING", "NOMINAL", "DEGRADED");
        decisionValid = builder.decisionValid;
        if (decisionValid != contextualRiskValid) {
            throw new IllegalArgumentException(
                    "health.decision_valid must match validity.contextual_risk");
        }
        staleOrInvalidComponents = immutableShortList(
                builder.staleOrInvalidComponents, "health.stale_or_invalid_components", 16);
        if ("NOMINAL".equals(healthMode) && !staleOrInvalidComponents.isEmpty()) {
            throw new IllegalArgumentException("NOMINAL health cannot contain invalid components");
        }
        if (c3Valid != Double.isFinite(c3SafeScorePct)) {
            throw new IllegalArgumentException(
                    "validity.c3 must match c3.safe_score_estimate_pct availability");
        }
        if (contextualRiskValid != Double.isFinite(contextRiskPct)) {
            throw new IllegalArgumentException(
                    "validity.contextual_risk must match contextual_risk.score_pct availability");
        }
    }

    public boolean allInputsValid() {
        return egoValid && frontCameraValid && driverCameraValid && c1Valid && c2Valid
                && c3Valid && driveQualityValid && contextualRiskValid;
    }

    public static Builder builder() {
        return new Builder();
    }

    private static long requireNonNegative(long value, String field) {
        if (value < 0L) {
            throw new IllegalArgumentException(field + " must be non-negative");
        }
        return value;
    }

    private static double requirePercent(double value, String field) {
        if (!Double.isFinite(value) || value < 0.0 || value > 100.0) {
            throw new IllegalArgumentException(field + " must be finite in [0, 100]");
        }
        return value;
    }

    private static double optionalPercent(double value, String field) {
        return Double.isNaN(value) ? value : requirePercent(value, field);
    }

    private static String requireText(String value, String field, int maximum) {
        String result = value == null ? "" : value.trim();
        if (result.isEmpty()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
        if (result.length() > maximum) {
            throw new IllegalArgumentException(field + " exceeds " + maximum + " characters");
        }
        for (int index = 0; index < result.length(); index++) {
            if (Character.isISOControl(result.charAt(index))) {
                throw new IllegalArgumentException(field + " contains control characters");
            }
        }
        return result;
    }

    private static String requireOneOf(String value, String field, String... allowed) {
        String result = requireText(value, field, 64);
        for (String candidate : allowed) {
            if (candidate.equals(result)) {
                return result;
            }
        }
        throw new IllegalArgumentException("unsupported " + field + ": " + result);
    }

    private static List<String> immutableShortList(List<String> values, String field, int maximum) {
        if (values == null || values.size() > maximum) {
            throw new IllegalArgumentException(field + " must contain at most " + maximum + " items");
        }
        List<String> copy = new ArrayList<>(values.size());
        for (String value : values) {
            copy.add(requireText(value, field + " item", 96));
        }
        return Collections.unmodifiableList(copy);
    }

    public static final class Builder {
        private String schemaVersion = SCHEMA;
        private SourceMode sourceMode;
        private String sessionId;
        private long sequence = -1;
        private long frameId = -1;
        private long sourceTimestampMs = -1;
        private long decisionTimestampMs = -1;
        private int ttlMs;
        private long expiresAtMs = -1;
        private boolean egoValid;
        private boolean frontCameraValid;
        private boolean driverCameraValid;
        private boolean c1Valid;
        private boolean c2Valid;
        private boolean c3Valid;
        private boolean driveQualityValid;
        private boolean contextualRiskValid;
        private double ttcSeconds = Double.POSITIVE_INFINITY;
        private boolean ttcValid;
        private double collisionProbabilityPct = Double.NaN;
        private boolean collisionWarning;
        private boolean c1ModelUpdated;
        private long c1ModelFrameId = -1;
        private long c1AgeMs = -1;
        private String driverState;
        private double driverConfidencePct = Double.NaN;
        private double attentionPct = Double.NaN;
        private double distractionPct = Double.NaN;
        private double fatiguePct = Double.NaN;
        private Boolean eyesOnRoad;
        private boolean driverWarning;
        private double c3SafeScorePct = Double.NaN;
        private String c3Grade;
        private String c3Scope;
        private String c3FormulaVersion;
        private boolean c3TailgatingPenaltyOmitted;
        private boolean driveQualityAvailable;
        private double driveQualityPct = Double.NaN;
        private String driveQualityGrade;
        private String driveQualityScope;
        private boolean driveQualityWindowReady;
        private String driveQualityFormulaVersion;
        private double contextRiskPct = Double.NaN;
        private String riskLevel;
        private String action;
        private double brakeRequestPct = Double.NaN;
        private List<String> riskReasons = Collections.emptyList();
        private boolean actuationAuthorized;
        private String healthMode;
        private boolean decisionValid;
        private List<String> staleOrInvalidComponents = Collections.emptyList();

        public Builder schemaVersion(String value) { schemaVersion = value; return this; }
        public Builder sourceMode(SourceMode value) { sourceMode = value; return this; }
        public Builder sessionId(String value) { sessionId = value; return this; }
        public Builder sequence(long value) { sequence = value; return this; }
        public Builder frameId(long value) { frameId = value; return this; }
        public Builder sourceTimestampMs(long value) { sourceTimestampMs = value; return this; }
        public Builder decisionTimestampMs(long value) { decisionTimestampMs = value; return this; }
        public Builder ttlMs(int value) { ttlMs = value; return this; }
        public Builder expiresAtMs(long value) { expiresAtMs = value; return this; }
        public Builder validity(boolean ego, boolean front, boolean driver, boolean c1,
                boolean c2, boolean c3, boolean quality, boolean risk) {
            egoValid = ego; frontCameraValid = front; driverCameraValid = driver;
            c1Valid = c1; c2Valid = c2; c3Valid = c3;
            driveQualityValid = quality; contextualRiskValid = risk; return this;
        }
        public Builder c1(double ttc, boolean ttcIsValid, double probability, boolean warning,
                boolean updated, long modelFrame, long age) {
            ttcSeconds = ttc; ttcValid = ttcIsValid; collisionProbabilityPct = probability;
            collisionWarning = warning; c1ModelUpdated = updated;
            c1ModelFrameId = modelFrame; c1AgeMs = age; return this;
        }
        public Builder c2(String state, double confidence, double attention, double distraction,
                double fatigue, Boolean eyes, boolean warning) {
            driverState = state; driverConfidencePct = confidence; attentionPct = attention;
            distractionPct = distraction; fatiguePct = fatigue;
            eyesOnRoad = eyes; driverWarning = warning; return this;
        }
        public Builder c3(double score, String grade, String scope, String formula,
                boolean tailgatingOmitted) {
            c3SafeScorePct = score; c3Grade = grade; c3Scope = scope;
            c3FormulaVersion = formula; c3TailgatingPenaltyOmitted = tailgatingOmitted; return this;
        }
        public Builder driveQuality(boolean available, double score, String grade, String scope,
                boolean windowReady, String formula) {
            driveQualityAvailable = available; driveQualityPct = score;
            driveQualityGrade = grade; driveQualityScope = scope;
            driveQualityWindowReady = windowReady; driveQualityFormulaVersion = formula; return this;
        }
        public Builder contextualRisk(double score, String level, String riskAction, double brake,
                List<String> reasons, boolean authorized) {
            contextRiskPct = score; riskLevel = level; action = riskAction;
            brakeRequestPct = brake; riskReasons = reasons; actuationAuthorized = authorized; return this;
        }
        public Builder health(String mode, boolean valid, List<String> invalid) {
            healthMode = mode; decisionValid = valid; staleOrInvalidComponents = invalid; return this;
        }
        public DecisionSnapshot build() { return new DecisionSnapshot(this); }
    }
}
