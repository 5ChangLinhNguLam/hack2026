package com.fptautomotive.safeloop;

/** Lifecycle contract shared by the mutually-exclusive UDP and HTTPS transports. */
public interface DecisionReceiver extends AutoCloseable {
    interface Listener {
        void onPacket(long generationToken, DecisionSnapshot packet, long receivedElapsedMs);
        void onMalformedPacket(long generationToken, String message);
        void onTransportError(long generationToken, String message);
    }

    long start();
    void stop();
    boolean isGenerationActive(long generationToken);

    @Override
    void close();
}
