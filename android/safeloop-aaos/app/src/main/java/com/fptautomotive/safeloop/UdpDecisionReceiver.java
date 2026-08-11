package com.fptautomotive.safeloop;

import android.os.SystemClock;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetSocketAddress;
import java.net.SocketException;
import java.net.SocketTimeoutException;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;

/** One-thread UDP listener for full, self-healing SafeLoop snapshots. */
public final class UdpDecisionReceiver implements AutoCloseable {
    public static final int DEFAULT_PORT = 48100;

    public interface Listener {
        void onPacket(
                long generationToken,
                DecisionSnapshot packet,
                long receivedElapsedMs);
        void onMalformedPacket(long generationToken, String message);
        void onTransportError(long generationToken, String message);
    }

    interface SocketFactory {
        DatagramSocket open() throws SocketException;
    }

    interface ElapsedClock {
        long nowMs();
    }

    private final int port;
    private final DecisionPacketParser parser;
    private final Listener listener;
    private final SocketFactory socketFactory;
    private final ElapsedClock elapsedClock;
    private boolean running;
    private long generation;
    private DatagramSocket socket;
    private Thread thread;

    public UdpDecisionReceiver(int port, DecisionPacketParser parser, Listener listener) {
        this(port, parser, listener, () -> new DatagramSocket(null),
                SystemClock::elapsedRealtime);
    }

    UdpDecisionReceiver(
            int port,
            DecisionPacketParser parser,
            Listener listener,
            SocketFactory socketFactory,
            ElapsedClock elapsedClock) {
        if (port < 1024 || port > 65535) {
            throw new IllegalArgumentException("UDP port must be in [1024, 65535]");
        }
        if (parser == null || listener == null || socketFactory == null || elapsedClock == null) {
            throw new IllegalArgumentException(
                    "parser, listener, socketFactory and elapsedClock are required");
        }
        this.port = port;
        this.parser = parser;
        this.listener = listener;
        this.socketFactory = socketFactory;
        this.elapsedClock = elapsedClock;
    }

    /** Starts reception and returns the token carried by every callback from this run. */
    public synchronized long start() {
        if (running) {
            return generation;
        }
        running = true;
        long activeGeneration = ++generation;
        thread = new Thread(
                () -> receiveLoop(activeGeneration),
                "safeloop-udp-receiver");
        thread.setDaemon(true);
        thread.start();
        return activeGeneration;
    }

    private void receiveLoop(long activeGeneration) {
        byte[] buffer = new byte[DecisionPacketParser.MAX_PACKET_BYTES + 1];
        DatagramSocket activeSocket = null;
        try {
            activeSocket = socketFactory.open();
            if (!claimSocket(activeGeneration, activeSocket)) {
                activeSocket.close();
                return;
            }
            activeSocket.setReuseAddress(true);
            activeSocket.bind(new InetSocketAddress(port));
            activeSocket.setSoTimeout(500);
            while (isActive(activeGeneration)) {
                DatagramPacket datagram = new DatagramPacket(buffer, buffer.length);
                try {
                    activeSocket.receive(datagram);
                    if (datagram.getLength() > DecisionPacketParser.MAX_PACKET_BYTES) {
                        reportMalformed(activeGeneration, "UDP packet exceeds maximum size");
                        continue;
                    }
                    try {
                        String payload = decodeUtf8(datagram);
                        DecisionSnapshot packet = parser.parse(payload);
                        reportPacket(activeGeneration, packet, elapsedClock.nowMs());
                    } catch (CharacterCodingException error) {
                        reportMalformed(activeGeneration, "UDP packet is not valid UTF-8");
                    } catch (DecisionPacketParser.PacketFormatException error) {
                        reportMalformed(activeGeneration, error.getMessage());
                    }
                } catch (SocketTimeoutException ignored) {
                    // Timeout exists solely so stop() is observed promptly.
                }
            }
        } catch (SocketException error) {
            if (isActive(activeGeneration)) {
                reportTransportError(
                        activeGeneration,
                        "Unable to bind UDP " + port + ": " + error.getMessage());
            }
        } catch (IOException error) {
            if (isActive(activeGeneration)) {
                reportTransportError(
                        activeGeneration, "UDP receive failed: " + error.getMessage());
            }
        } finally {
            if (activeSocket != null) {
                activeSocket.close();
            }
            synchronized (this) {
                if (socket == activeSocket) {
                    socket = null;
                }
                if (generation == activeGeneration) {
                    running = false;
                    if (thread == Thread.currentThread()) {
                        thread = null;
                    }
                }
            }
        }
    }

    private static String decodeUtf8(DatagramPacket datagram)
            throws CharacterCodingException {
        return StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(
                        datagram.getData(), datagram.getOffset(), datagram.getLength()))
                .toString();
    }

    private synchronized boolean claimSocket(
            long activeGeneration, DatagramSocket candidate) {
        if (!running || generation != activeGeneration) {
            return false;
        }
        socket = candidate;
        return true;
    }

    /** True only while callbacks carrying {@code generationToken} remain current. */
    public synchronized boolean isGenerationActive(long generationToken) {
        return generationToken > 0L && generation == generationToken;
    }

    private boolean isActive(long activeGeneration) {
        return isGenerationActive(activeGeneration);
    }

    private void reportPacket(
            long activeGeneration,
            DecisionSnapshot packet,
            long receivedElapsedMs) {
        if (isActive(activeGeneration)) {
            listener.onPacket(activeGeneration, packet, receivedElapsedMs);
        }
    }

    private void reportMalformed(long activeGeneration, String message) {
        if (isActive(activeGeneration)) {
            listener.onMalformedPacket(activeGeneration, message);
        }
    }

    private void reportTransportError(long activeGeneration, String message) {
        if (isActive(activeGeneration)) {
            listener.onTransportError(activeGeneration, message);
        }
    }

    public synchronized void stop() {
        running = false;
        generation++;
        DatagramSocket activeSocket = socket;
        socket = null;
        if (activeSocket != null) {
            activeSocket.close();
        }
        Thread activeThread = thread;
        if (activeThread != null) {
            activeThread.interrupt();
        }
        thread = null;
    }

    @Override
    public void close() {
        stop();
    }
}
