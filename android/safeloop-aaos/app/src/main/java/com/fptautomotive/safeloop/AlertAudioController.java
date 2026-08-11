package com.fptautomotive.safeloop;

import android.media.AudioManager;
import android.media.ToneGenerator;
import android.util.Log;

/** Rate-limited audio cues.  STALE/NO_DATA decisions are always silent. */
public final class AlertAudioController implements AutoCloseable {
    private static final String TAG = "SafeLoopAudio";
    private static final long HIGH_REPEAT_MS = 2500L;
    private static final long CRITICAL_REPEAT_MS = 1000L;
    private final ToneGenerator tones;
    private AlertPolicy.AudioCue previous = AlertPolicy.AudioCue.NONE;
    private AlertPolicy.AudioCue lastPlayedCue = AlertPolicy.AudioCue.NONE;
    private long lastPlayedMs = Long.MIN_VALUE;
    private boolean enabled;

    public AlertAudioController() {
        ToneGenerator candidate;
        try {
            candidate = new ToneGenerator(AudioManager.STREAM_ALARM, 75);
        } catch (RuntimeException error) {
            Log.e(TAG, "Audio cue initialization failed", error);
            candidate = null;
        }
        tones = candidate;
    }

    public void setEnabled(boolean value) {
        enabled = value;
        if (!enabled && tones != null) {
            tones.stopTone();
            previous = AlertPolicy.AudioCue.NONE;
        }
    }

    public void apply(AlertPolicy.Decision decision, long nowElapsedMs) {
        if (!enabled || tones == null || decision == null) {
            previous = AlertPolicy.AudioCue.NONE;
            return;
        }
        if (decision.audioCue == AlertPolicy.AudioCue.NONE) {
            if (previous != AlertPolicy.AudioCue.NONE) {
                tones.stopTone();
            }
            previous = AlertPolicy.AudioCue.NONE;
            return;
        }
        long repeat = decision.audioCue == AlertPolicy.AudioCue.CRITICAL
                ? CRITICAL_REPEAT_MS
                : HIGH_REPEAT_MS;
        boolean escalated = decision.audioCue.ordinal() > lastPlayedCue.ordinal();
        boolean due = lastPlayedMs == Long.MIN_VALUE || nowElapsedMs - lastPlayedMs >= repeat;
        if (escalated || due) {
            int tone = decision.audioCue == AlertPolicy.AudioCue.CRITICAL
                    ? ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD
                    : ToneGenerator.TONE_PROP_BEEP2;
            int duration = decision.audioCue == AlertPolicy.AudioCue.CRITICAL ? 350 : 220;
            try {
                if (tones.startTone(tone, duration)) {
                    lastPlayedMs = nowElapsedMs;
                    lastPlayedCue = decision.audioCue;
                }
            } catch (RuntimeException error) {
                Log.e(TAG, "Audio cue playback failed", error);
            }
        }
        previous = decision.audioCue;
    }

    @Override
    public void close() {
        if (tones != null) {
            tones.stopTone();
            tones.release();
        }
    }
}
