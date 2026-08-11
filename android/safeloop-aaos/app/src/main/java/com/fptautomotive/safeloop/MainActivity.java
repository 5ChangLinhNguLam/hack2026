package com.fptautomotive.safeloop;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.view.WindowManager;

/** Single-screen native AAOS dashboard; no WebView and no embedded replay. */
public final class MainActivity extends Activity implements UdpDecisionReceiver.Listener {
    public static final String EXTRA_UDP_PORT = "udp_port";
    private static final long UI_TICK_MS = 250L;

    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final DecisionStateMachine stateMachine = new DecisionStateMachine();
    private final AlertPolicy alertPolicy = new AlertPolicy();
    private DashboardView dashboard;
    private UdpDecisionReceiver receiver;
    private AlertAudioController audio;
    private boolean started;
    private boolean destroyed;

    private final Runnable ticker = new Runnable() {
        @Override
        public void run() {
            if (!started) {
                return;
            }
            render(SystemClock.elapsedRealtime());
            mainHandler.postDelayed(this, UI_TICK_MS);
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        dashboard = new DashboardView(this);
        setContentView(dashboard);
        audio = new AlertAudioController();
        int requestedPort = getIntent().getIntExtra(
                EXTRA_UDP_PORT, UdpDecisionReceiver.DEFAULT_PORT);
        if (requestedPort < 1024 || requestedPort > 65535) {
            requestedPort = UdpDecisionReceiver.DEFAULT_PORT;
        }
        receiver = new UdpDecisionReceiver(requestedPort, new DecisionPacketParser(), this);
        render(SystemClock.elapsedRealtime());
    }

    @Override
    protected void onStart() {
        super.onStart();
        started = true;
        receiver.start();
        mainHandler.post(ticker);
    }

    @Override
    protected void onResume() {
        super.onResume();
        audio.setEnabled(true);
    }

    @Override
    protected void onPause() {
        audio.setEnabled(false);
        super.onPause();
    }

    @Override
    protected void onStop() {
        started = false;
        mainHandler.removeCallbacks(ticker);
        receiver.stop();
        super.onStop();
    }

    @Override
    protected void onDestroy() {
        destroyed = true;
        mainHandler.removeCallbacksAndMessages(null);
        receiver.close();
        audio.close();
        super.onDestroy();
    }

    @Override
    public void onPacket(
            long generationToken,
            DecisionSnapshot packet,
            long receivedElapsedMs) {
        mainHandler.post(() -> {
            if (acceptsReceiverCallback(generationToken)) {
                stateMachine.accept(packet, receivedElapsedMs);
                render(SystemClock.elapsedRealtime());
            }
        });
    }

    @Override
    public void onMalformedPacket(long generationToken, String message) {
        mainHandler.post(() -> {
            if (acceptsReceiverCallback(generationToken)) {
                stateMachine.recordParseError(message);
                render(SystemClock.elapsedRealtime());
            }
        });
    }

    @Override
    public void onTransportError(long generationToken, String message) {
        mainHandler.post(() -> {
            if (acceptsReceiverCallback(generationToken)) {
                stateMachine.recordTransportError(message);
                render(SystemClock.elapsedRealtime());
            }
        });
    }

    private boolean acceptsReceiverCallback(long generationToken) {
        return started && !destroyed && receiver != null
                && receiver.isGenerationActive(generationToken);
    }

    private void render(long nowElapsedMs) {
        DashboardState state = stateMachine.stateAt(nowElapsedMs);
        AlertPolicy.Decision alert = alertPolicy.evaluate(state);
        dashboard.setDashboardState(state, alert);
        audio.apply(alert, nowElapsedMs);
    }
}
