package com.fptautomotive.safeloop;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.util.AttributeSet;
import android.view.View;

import java.util.Locale;

/** Native, glanceable AAOS dashboard rendered entirely with View/Canvas. */
public final class DashboardView extends View {
    private static final int BACKGROUND = Color.rgb(5, 17, 24);
    private static final int PANEL = Color.rgb(9, 29, 38);
    private static final int PANEL_EDGE = Color.rgb(24, 59, 70);
    private static final int TEXT = Color.rgb(240, 248, 250);
    private static final int MUTED = Color.rgb(132, 157, 166);
    private static final int CYAN = Color.rgb(70, 218, 219);
    private static final int GREEN = Color.rgb(157, 224, 70);
    private static final int AMBER = Color.rgb(255, 183, 55);
    private static final int RED = Color.rgb(255, 91, 97);

    private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF rect = new RectF();
    private final Path path = new Path();
    private final float density;
    private final float scaledDensity;
    private DashboardState state = new DashboardState(
            DashboardState.Health.NO_DATA, null, -1L, 0L, 0L, 0L, 0L, "");
    private AlertPolicy.Decision alert = new AlertPolicy().evaluate(state);
    private boolean cloudTransport;

    public DashboardView(Context context) {
        this(context, null);
    }

    public DashboardView(Context context, AttributeSet attributes) {
        this(context, attributes, 0);
    }

    public DashboardView(Context context, AttributeSet attributes, int defaultStyle) {
        super(context, attributes, defaultStyle);
        density = getResources().getDisplayMetrics().density;
        scaledDensity = getResources().getDisplayMetrics().scaledDensity;
        paint.setTypeface(Typeface.create("sans-serif", Typeface.NORMAL));
        setFocusable(false);
        setImportantForAccessibility(IMPORTANT_FOR_ACCESSIBILITY_YES);
        updateAccessibilityDescription();
    }

    public void setDashboardState(DashboardState newState, AlertPolicy.Decision newAlert) {
        if (newState == null || newAlert == null) {
            throw new IllegalArgumentException("state and alert are required");
        }
        state = newState;
        alert = newAlert;
        updateAccessibilityDescription();
        invalidate();
    }

    public void setCloudTransport(boolean enabled) {
        cloudTransport = enabled;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        canvas.drawColor(BACKGROUND);
        float width = getWidth();
        float height = getHeight();
        float margin = Math.min(dp(24), width * 0.025f);
        float gap = Math.min(dp(16), width * 0.015f);
        float headerHeight = Math.min(dp(70), height * 0.12f);
        float alertHeight = Math.min(dp(54), height * 0.09f);

        drawHeader(canvas, margin, margin, width - margin * 2f, headerHeight);
        float alertY = margin + headerHeight + gap;
        drawAlertStrip(canvas, margin, alertY, width - margin * 2f, alertHeight);

        float contentY = alertY + alertHeight + gap;
        float contentHeight = Math.max(0f, height - contentY - margin);
        float availableWidth = width - margin * 2f - gap * 2f;
        float leftWidth = availableWidth * 0.29f;
        float centerWidth = availableWidth * 0.39f;
        float rightWidth = availableWidth - leftWidth - centerWidth;
        float leftX = margin;
        float centerX = leftX + leftWidth + gap;
        float rightX = centerX + centerWidth + gap;

        float halfHeight = (contentHeight - gap) / 2f;
        drawCollisionPanel(canvas, leftX, contentY, leftWidth, halfHeight);
        drawRiskPanel(canvas, leftX, contentY + halfHeight + gap, leftWidth, halfHeight);
        drawRoadPanel(canvas, centerX, contentY, centerWidth, contentHeight);

        float driverHeight = Math.max(contentHeight * 0.60f, dp(170));
        driverHeight = Math.min(driverHeight, contentHeight - gap - dp(92));
        drawDriverPanel(canvas, rightX, contentY, rightWidth, driverHeight);
        float scoreY = contentY + driverHeight + gap;
        float scoreHeight = contentHeight - driverHeight - gap;
        drawScorePanel(canvas, rightX, scoreY, rightWidth, scoreHeight);

        if (!hasValidRiskDecision()) {
            drawUnavailableOverlay(canvas, centerX, contentY, centerWidth, contentHeight);
        }
    }

    private void drawHeader(Canvas canvas, float x, float y, float width, float height) {
        setPaint(CYAN, Paint.Style.STROKE, dp(2));
        rect.set(x, y + dp(8), x + dp(42), y + dp(50));
        canvas.drawRoundRect(rect, dp(10), dp(10), paint);
        drawText(canvas, "SL", x + dp(13), y + dp(36), sp(15), CYAN, true);
        drawText(canvas, "SAFELOOP", x + dp(58), y + dp(28), sp(23), TEXT, true);
        String provenance = cloudTransport
                ? cloudProvenance()
                : "TRANSPORT UDP · ROOM-LOCAL DECISION";
        drawText(canvas, provenance, x + dp(58), y + dp(50), sp(9), MUTED, false);

        String source = cloudTransport ? "TRANSPORT AWS"
                : (state.snapshot == null ? "SOURCE —" : "SOURCE " + state.snapshot.sourceMode);
        int sourceColor = state.snapshot != null
                && state.snapshot.sourceMode == DecisionSnapshot.SourceMode.LIVE
                ? GREEN : AMBER;
        float badgeWidth = dp(118);
        drawBadge(canvas, source, x + width - badgeWidth * 2f - dp(14), y + dp(10),
                badgeWidth, dp(34), sourceColor);
        int healthColor = healthColor(state.health);
        drawBadge(canvas, state.health.name(), x + width - badgeWidth, y + dp(10), badgeWidth, dp(34), healthColor);
        if (state.packetAgeMs >= 0L) {
            long badPackets = state.rejectedPackets + state.parseErrors;
            drawEllipsizedText(canvas,
                    "seq " + state.snapshot.sequence + "  drop " + state.droppedPackets
                            + "  bad " + badPackets,
                    x + width - badgeWidth * 2f - dp(14), y + height - dp(5),
                    badgeWidth, sp(10), MUTED, false);
            drawText(canvas, "age " + state.packetAgeMs + " ms", x + width - badgeWidth,
                    y + height - dp(5), sp(10), MUTED, false);
        }
    }

    private void drawAlertStrip(Canvas canvas, float x, float y, float width, float height) {
        int color = severityColor(alert.severity);
        setPaint(withAlpha(color, 34), Paint.Style.FILL, 0f);
        rect.set(x, y, x + width, y + height);
        canvas.drawRoundRect(rect, dp(12), dp(12), paint);
        setPaint(color, Paint.Style.STROKE, dp(1));
        canvas.drawRoundRect(rect, dp(12), dp(12), paint);
        drawText(canvas, alert.label, x + dp(18), centerBaseline(y, height, sp(18)), sp(18), color, true);
        drawEllipsizedText(canvas, alert.detail, x + dp(190), centerBaseline(y, height, sp(13)),
                width - dp(208), sp(13), TEXT, false);
        if (cloudTransport && state.snapshot != null
                && state.snapshot.sourceMode == DecisionSnapshot.SourceMode.REPLAY) {
            drawTextRight(canvas, "RECORDED STREAM — LIVE MODEL INFERENCE",
                    x + width - dp(14), y + dp(15), sp(8), AMBER, true);
        }
    }

    private String cloudProvenance() {
        if (state.snapshot == null) {
            return "CAMERA — · TELEMETRY — · INFERENCE LIVE_MODEL";
        }
        if (state.snapshot.sourceMode == DecisionSnapshot.SourceMode.REPLAY) {
            return "CAMERA RECORDED · TELEMETRY RECORDED_DATA · INFERENCE LIVE_MODEL";
        }
        return "CAMERA LIVE · TELEMETRY THIRD_PARTY · INFERENCE LIVE_MODEL";
    }

    private void drawCollisionPanel(Canvas canvas, float x, float y, float width, float height) {
        drawPanel(canvas, x, y, width, height);
        float px = x + dp(18);
        drawSectionTitle(canvas, "C1 · COLLISION INTELLIGENCE", px, y + dp(27));
        drawText(canvas, "TIME TO COLLISION", px, y + dp(55), sp(13), MUTED, false);
        if (hasC1Decision()) {
            DecisionSnapshot packet = state.snapshot;
            String ttc = Double.isFinite(packet.ttcSeconds)
                    ? String.format(Locale.US, "%.2f s", packet.ttcSeconds)
                    : "∞";
            drawText(canvas, ttc, px, y + Math.min(height - dp(48), dp(111)), sp(38), TEXT, true);
            String confidence = Double.isFinite(packet.collisionProbabilityPct)
                    ? String.format(Locale.US, "COLLISION PROB. %.0f%%",
                            packet.collisionProbabilityPct)
                    : "COLLISION PROB. —";
            drawText(canvas, confidence, px, y + height - dp(23), sp(12), CYAN, true);
        } else {
            drawText(canvas, "—", px, y + Math.min(height - dp(48), dp(111)), sp(38), MUTED, true);
            drawText(canvas, "INPUT UNAVAILABLE", px, y + height - dp(23), sp(12), MUTED, true);
        }
    }

    private void drawRiskPanel(Canvas canvas, float x, float y, float width, float height) {
        drawPanel(canvas, x, y, width, height);
        float px = x + dp(18);
        drawSectionTitle(canvas, "FUSION · DECISION POLICY", px, y + dp(27));
        drawText(canvas, "CONTEXT RISK", px, y + dp(55), sp(13), MUTED, false);
        if (hasValidRiskDecision()) {
            DecisionSnapshot packet = state.snapshot;
            int riskColor = scoreRiskColor(packet.contextRiskPct);
            drawText(canvas, formatPercent(packet.contextRiskPct), px, y + dp(99), sp(34), riskColor, true);
            drawBar(canvas, px, y + dp(112), width - dp(36), packet.contextRiskPct, riskColor);
            drawEllipsizedText(canvas, packet.action.replace('_', ' '), px,
                    y + height - dp(22), width - dp(36), sp(12), TEXT, true);
        } else {
            drawText(canvas, "—", px, y + dp(99), sp(34), MUTED, true);
        }
    }

    private void drawRoadPanel(Canvas canvas, float x, float y, float width, float height) {
        drawPanel(canvas, x, y, width, height);
        setPaint(Color.rgb(17, 42, 51), Paint.Style.FILL, 0f);
        path.reset();
        path.moveTo(x + width * 0.34f, y + dp(20));
        path.lineTo(x + width * 0.08f, y + height - dp(18));
        path.lineTo(x + width * 0.92f, y + height - dp(18));
        path.lineTo(x + width * 0.66f, y + dp(20));
        path.close();
        canvas.drawPath(path, paint);
        setPaint(MUTED, Paint.Style.STROKE, dp(2));
        paint.setStrokeWidth(dp(2));
        for (int index = 0; index < 4; index++) {
            float top = y + dp(35) + index * (height - dp(70)) / 4f;
            float bottom = top + Math.min(dp(28), height * 0.07f);
            float topHalf = width * (0.03f + index * 0.012f);
            canvas.drawLine(x + width / 2f - topHalf, top,
                    x + width / 2f - topHalf * 1.25f, bottom, paint);
            canvas.drawLine(x + width / 2f + topHalf, top,
                    x + width / 2f + topHalf * 1.25f, bottom, paint);
        }

        float egoWidth = Math.min(dp(78), width * 0.20f);
        float egoHeight = Math.min(dp(116), height * 0.24f);
        float egoY = y + height - egoHeight - dp(34);
        drawVehicle(canvas, x + width / 2f, egoY, egoWidth, egoHeight, CYAN, "EGO");

        if (hasC1Decision() && state.snapshot.ttcValid
                && Double.isFinite(state.snapshot.ttcSeconds)) {
            double normalized = Math.max(0.0, Math.min(1.0, state.snapshot.ttcSeconds / 6.0));
            float targetY = egoY - dp(70) - (float) normalized * Math.max(dp(30), height * 0.28f);
            int targetColor = severityColor(alert.severity);
            drawVehicle(canvas, x + width / 2f, targetY,
                    egoWidth * 0.78f, egoHeight * 0.70f, targetColor,
                    String.format(Locale.US, "%.1fs", state.snapshot.ttcSeconds));
        }
    }

    private void drawDriverPanel(Canvas canvas, float x, float y, float width, float height) {
        drawPanel(canvas, x, y, width, height);
        float px = x + dp(18);
        drawSectionTitle(canvas, "C2 · DRIVER INTELLIGENCE", px, y + dp(27));
        if (!hasC2Decision()) {
            drawText(canvas, "—", px, y + dp(68), sp(25), MUTED, true);
            return;
        }
        DecisionSnapshot packet = state.snapshot;
        drawEllipsizedText(canvas, packet.driverState.replace('_', ' '), px, y + dp(66),
                width - dp(36), sp(23), TEXT, true);
        float firstBar = y + Math.min(dp(94), height * 0.48f);
        float barStep = Math.max(dp(20), (height - (firstBar - y) - dp(14)) / 3f);
        drawMetricBar(canvas, "ATTENTION", packet.attentionPct, px, firstBar, width - dp(36), CYAN);
        drawMetricBar(canvas, "DISTRACTION", packet.distractionPct, px,
                firstBar + barStep, width - dp(36), AMBER);
        drawMetricBar(canvas, "FATIGUE", packet.fatiguePct, px,
                firstBar + barStep * 2f, width - dp(36), RED);
    }

    private void drawScorePanel(Canvas canvas, float x, float y, float width, float height) {
        drawPanel(canvas, x, y, width, height);
        float mid = x + width / 2f;
        setPaint(PANEL_EDGE, Paint.Style.STROKE, dp(1));
        canvas.drawLine(mid, y + dp(14), mid, y + height - dp(14), paint);
        drawSectionTitle(canvas, "C3 · SAFE EST.", x + dp(16), y + dp(26));
        drawSectionTitle(canvas, "DRIVE QUALITY", mid + dp(16), y + dp(26));
        boolean c3Available = hasFreshPacket() && state.snapshot.c3Valid
                && Double.isFinite(state.snapshot.c3SafeScorePct);
        boolean qualityAvailable = hasFreshPacket() && state.snapshot.driveQualityValid
                && state.snapshot.driveQualityAvailable
                && Double.isFinite(state.snapshot.driveQualityPct);
        if (hasFreshPacket()) {
            String c3Value = c3Available
                    ? formatPercent(state.snapshot.c3SafeScorePct)
                    : "—";
            String qualityValue = qualityAvailable
                    ? formatPercent(state.snapshot.driveQualityPct)
                    : "—";
            drawCenteredText(canvas, c3Value,
                    x, y + dp(38), width / 2f, Math.max(dp(34), height - dp(48)),
                    sp(29), c3Available
                            ? scoreQualityColor(state.snapshot.c3SafeScorePct) : MUTED, true);
            drawCenteredText(canvas, qualityValue,
                    mid, y + dp(38), width / 2f, Math.max(dp(34), height - dp(48)),
                    sp(29), qualityAvailable
                            ? scoreQualityColor(state.snapshot.driveQualityPct) : MUTED, true);
            String c3Meta = c3Available
                    ? state.snapshot.c3Grade + " · " + state.snapshot.c3Scope
                    : "N/A · " + state.snapshot.c3Scope;
            String qualityMeta = qualityAvailable
                    ? state.snapshot.driveQualityGrade + " · " + state.snapshot.driveQualityScope
                    : "N/A · " + state.snapshot.driveQualityScope;
            drawCenteredText(canvas, c3Meta, x, y + height - dp(31), width / 2f, dp(24),
                    sp(10), MUTED, true);
            drawCenteredText(canvas, qualityMeta, mid, y + height - dp(31), width / 2f, dp(24),
                    sp(10), MUTED, true);
        } else {
            drawCenteredText(canvas, "—", x, y + dp(38), width / 2f,
                    Math.max(dp(34), height - dp(48)), sp(29), MUTED, true);
            drawCenteredText(canvas, "—", mid, y + dp(38), width / 2f,
                    Math.max(dp(34), height - dp(48)), sp(29), MUTED, true);
        }
    }

    private void drawUnavailableOverlay(Canvas canvas, float x, float y, float width, float height) {
        setPaint(withAlpha(BACKGROUND, 218), Paint.Style.FILL, 0f);
        rect.set(x + dp(1), y + dp(1), x + width - dp(1), y + height - dp(1));
        canvas.drawRoundRect(rect, dp(14), dp(14), paint);
        int color = healthColor(state.health);
        drawCenteredText(canvas, alert.label, x, y + height * 0.35f, width, height * 0.18f,
                sp(29), color, true);
        drawCenteredText(canvas, alert.detail, x + dp(20), y + height * 0.53f,
                width - dp(40), height * 0.14f, sp(13), TEXT, false);
        String stats = "drop " + state.droppedPackets + "  reject " + state.rejectedPackets
                + "  malformed " + state.parseErrors;
        drawCenteredText(canvas, stats, x + dp(20), y + height * 0.67f,
                width - dp(40), height * 0.10f, sp(11), MUTED, false);
    }

    private boolean hasFreshPacket() {
        return (state.health == DashboardState.Health.LIVE
                || state.health == DashboardState.Health.DEGRADED)
                && state.snapshot != null;
    }

    private boolean hasValidRiskDecision() {
        return hasFreshPacket()
                && state.snapshot.contextualRiskValid
                && state.snapshot.decisionValid;
    }

    private boolean hasC1Decision() {
        return hasFreshPacket() && state.snapshot.c1Valid;
    }

    private boolean hasC2Decision() {
        return hasFreshPacket() && state.snapshot.c2Valid;
    }

    private void drawPanel(Canvas canvas, float x, float y, float width, float height) {
        rect.set(x, y, x + width, y + height);
        setPaint(PANEL, Paint.Style.FILL, 0f);
        canvas.drawRoundRect(rect, dp(15), dp(15), paint);
        setPaint(PANEL_EDGE, Paint.Style.STROKE, dp(1));
        canvas.drawRoundRect(rect, dp(15), dp(15), paint);
    }

    private void drawSectionTitle(Canvas canvas, String value, float x, float baseline) {
        drawText(canvas, value, x, baseline, sp(11), CYAN, true);
    }

    private void drawMetricBar(
            Canvas canvas, String label, double value, float x, float y, float width, int color) {
        drawText(canvas, label, x, y, sp(10), MUTED, false);
        drawTextRight(canvas, String.format(Locale.US, "%.0f%%", value), x + width, y,
                sp(10), TEXT, true);
        drawBar(canvas, x, y + dp(9), width, value, color);
    }

    private void drawBar(Canvas canvas, float x, float y, float width, double value, int color) {
        float height = dp(6);
        rect.set(x, y, x + width, y + height);
        setPaint(Color.rgb(23, 50, 59), Paint.Style.FILL, 0f);
        canvas.drawRoundRect(rect, height / 2f, height / 2f, paint);
        rect.right = x + width * (float) Math.max(0.0, Math.min(100.0, value)) / 100f;
        setPaint(color, Paint.Style.FILL, 0f);
        canvas.drawRoundRect(rect, height / 2f, height / 2f, paint);
    }

    private void drawVehicle(
            Canvas canvas, float centerX, float top, float width, float height, int color, String label) {
        rect.set(centerX - width / 2f, top, centerX + width / 2f, top + height);
        setPaint(withAlpha(color, 30), Paint.Style.FILL, 0f);
        canvas.drawRoundRect(rect, width * 0.22f, width * 0.22f, paint);
        setPaint(color, Paint.Style.STROKE, dp(2));
        canvas.drawRoundRect(rect, width * 0.22f, width * 0.22f, paint);
        drawCenteredText(canvas, label, rect.left, rect.top, rect.width(), rect.height(),
                Math.min(sp(11), width * 0.22f), color, true);
    }

    private void drawBadge(
            Canvas canvas, String label, float x, float y, float width, float height, int color) {
        rect.set(x, y, x + width, y + height);
        setPaint(withAlpha(color, 22), Paint.Style.FILL, 0f);
        canvas.drawRoundRect(rect, height / 2f, height / 2f, paint);
        setPaint(withAlpha(color, 130), Paint.Style.STROKE, dp(1));
        canvas.drawRoundRect(rect, height / 2f, height / 2f, paint);
        drawCenteredText(canvas, label, x, y, width, height, sp(10), color, true);
    }

    private void drawCenteredText(
            Canvas canvas, String value, float x, float y, float width, float height,
            float size, int color, boolean bold) {
        setTextPaint(size, color, bold);
        String fitted = fit(value, width - dp(8));
        float baseline = y + (height - (paint.descent() + paint.ascent())) / 2f;
        canvas.drawText(fitted, x + (width - paint.measureText(fitted)) / 2f, baseline, paint);
    }

    private void drawEllipsizedText(
            Canvas canvas, String value, float x, float baseline, float maxWidth,
            float size, int color, boolean bold) {
        setTextPaint(size, color, bold);
        canvas.drawText(fit(value, maxWidth), x, baseline, paint);
    }

    private void drawText(
            Canvas canvas, String value, float x, float baseline, float size, int color, boolean bold) {
        setTextPaint(size, color, bold);
        canvas.drawText(value, x, baseline, paint);
    }

    private void drawTextRight(
            Canvas canvas, String value, float right, float baseline, float size, int color, boolean bold) {
        setTextPaint(size, color, bold);
        canvas.drawText(value, right - paint.measureText(value), baseline, paint);
    }

    private String fit(String value, float maxWidth) {
        String safe = value == null || value.isEmpty() ? "—" : value;
        if (paint.measureText(safe) <= maxWidth) {
            return safe;
        }
        String suffix = "…";
        int end = safe.length();
        while (end > 0 && paint.measureText(safe.substring(0, end) + suffix) > maxWidth) {
            end--;
        }
        return end == 0 ? suffix : safe.substring(0, end) + suffix;
    }

    private void setTextPaint(float size, int color, boolean bold) {
        paint.setStyle(Paint.Style.FILL);
        paint.setStrokeWidth(1f);
        paint.setColor(color);
        paint.setTextSize(size);
        paint.setTypeface(Typeface.create("sans-serif", bold ? Typeface.BOLD : Typeface.NORMAL));
    }

    private void setPaint(int color, Paint.Style style, float strokeWidth) {
        paint.setColor(color);
        paint.setStyle(style);
        paint.setStrokeWidth(strokeWidth);
    }

    private float centerBaseline(float y, float height, float textSize) {
        setTextPaint(textSize, TEXT, false);
        return y + (height - (paint.descent() + paint.ascent())) / 2f;
    }

    private int healthColor(DashboardState.Health health) {
        switch (health) {
            case LIVE:
                return GREEN;
            case DEGRADED:
                return AMBER;
            case STALE:
                return RED;
            case NO_DATA:
            default:
                return MUTED;
        }
    }

    private int severityColor(AlertPolicy.Severity severity) {
        switch (severity) {
            case CRITICAL:
                return RED;
            case HIGH:
            case CAUTION:
                return AMBER;
            case SAFE:
                return GREEN;
            case NONE:
            default:
                return MUTED;
        }
    }

    private int scoreRiskColor(double value) {
        return value >= 70.0 ? RED : value >= 40.0 ? AMBER : GREEN;
    }

    private int scoreQualityColor(double value) {
        return value >= 80.0 ? GREEN : value >= 60.0 ? AMBER : RED;
    }

    private static int withAlpha(int color, int alpha) {
        return Color.argb(alpha, Color.red(color), Color.green(color), Color.blue(color));
    }

    private static String formatPercent(double value) {
        return String.format(Locale.US, "%.0f%%", value);
    }

    private void updateAccessibilityDescription() {
        StringBuilder description = new StringBuilder("SafeLoop. ")
                .append(state.health).append(". ").append(alert.label).append(". ");
        if (hasFreshPacket()) {
            DecisionSnapshot packet = state.snapshot;
            description.append("Time to collision ").append(hasC1Decision()
                    && packet.ttcValid && Double.isFinite(packet.ttcSeconds)
                    ? String.format(Locale.US, "%.2f seconds", packet.ttcSeconds)
                    : "unavailable");
            description.append(". Driver state ")
                    .append(hasC2Decision() ? packet.driverState : "unavailable");
            description.append(". Context risk ")
                    .append(hasValidRiskDecision()
                            ? Math.round(packet.contextRiskPct) + " percent" : "unavailable");
            description.append(". C3 safe estimate ")
                    .append(packet.c3Valid && Double.isFinite(packet.c3SafeScorePct)
                            ? Math.round(packet.c3SafeScorePct) + " percent" : "unavailable");
            description.append(". Drive quality ")
                    .append(packet.driveQualityValid && packet.driveQualityAvailable
                            ? Math.round(packet.driveQualityPct) + " percent" : "unavailable")
                    .append('.');
        } else {
            description.append(alert.detail).append('.');
        }
        setContentDescription(description.toString());
    }

    private float dp(float value) {
        return value * density;
    }

    private float sp(float value) {
        return value * scaledDensity;
    }
}
