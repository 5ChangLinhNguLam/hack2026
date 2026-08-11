package com.fptautomotive.safeloop;

import org.junit.Test;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketAddress;
import java.net.SocketException;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

public final class UdpDecisionReceiverTest {
    @Test
    public void callbacksCarryGenerationSoQueuedOldWorkIsRejectedAfterRestart()
            throws Exception {
        List<Long> callbackTokens = new CopyOnWriteArrayList<>();
        CountDownLatch callbacks = new CountDownLatch(2);
        UdpDecisionReceiver.Listener listener = new UdpDecisionReceiver.Listener() {
            @Override
            public void onPacket(
                    long generationToken, DecisionSnapshot packet, long receivedElapsedMs) {
                callbackTokens.add(generationToken);
                callbacks.countDown();
            }

            @Override
            public void onMalformedPacket(long generationToken, String message) {
                throw new AssertionError(message);
            }

            @Override
            public void onTransportError(long generationToken, String message) {
                throw new AssertionError(message);
            }
        };
        UdpDecisionReceiver receiver = new UdpDecisionReceiver(
                48_100,
                new DecisionPacketParser(),
                listener,
                () -> new ControlledSocket(DecisionFixtures.validJson()),
                () -> 123L);

        long oldGeneration = receiver.start();
        awaitCount(callbacks, 1L);
        receiver.stop();
        long newGeneration = receiver.start();
        assertTrue(callbacks.await(2L, TimeUnit.SECONDS));

        assertEquals(2, callbackTokens.size());
        assertEquals(oldGeneration, callbackTokens.get(0).longValue());
        assertEquals(newGeneration, callbackTokens.get(1).longValue());
        assertFalse(receiver.isGenerationActive(oldGeneration));
        assertTrue(receiver.isGenerationActive(newGeneration));
        receiver.stop();
    }

    @Test
    public void delayedOldThreadCannotOverwriteRestartedSocket() throws Exception {
        ControlledSocket oldSocket = new ControlledSocket(null);
        ControlledSocket newSocket = new ControlledSocket(null);
        CountDownLatch firstOpenEntered = new CountDownLatch(1);
        CountDownLatch releaseFirstOpen = new CountDownLatch(1);
        AtomicInteger calls = new AtomicInteger();
        UdpDecisionReceiver.SocketFactory factory = () -> {
            if (calls.incrementAndGet() == 1) {
                firstOpenEntered.countDown();
                boolean interrupted = false;
                long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2L);
                while (releaseFirstOpen.getCount() != 0L && System.nanoTime() < deadline) {
                    try {
                        releaseFirstOpen.await(20L, TimeUnit.MILLISECONDS);
                    } catch (InterruptedException error) {
                        interrupted = true;
                    }
                }
                if (interrupted) {
                    Thread.currentThread().interrupt();
                }
                if (releaseFirstOpen.getCount() != 0L) {
                    throw new SocketException("timed out waiting to release old socket");
                }
                return oldSocket;
            }
            return newSocket;
        };
        UdpDecisionReceiver receiver = new UdpDecisionReceiver(
                48_100,
                new DecisionPacketParser(),
                new NoOpListener(),
                factory,
                () -> 123L);

        long oldGeneration = receiver.start();
        assertTrue(firstOpenEntered.await(2L, TimeUnit.SECONDS));
        receiver.stop();
        long newGeneration = receiver.start();
        assertTrue(newSocket.bound.await(2L, TimeUnit.SECONDS));
        releaseFirstOpen.countDown();
        assertTrue(oldSocket.closed.await(2L, TimeUnit.SECONDS));

        assertFalse(receiver.isGenerationActive(oldGeneration));
        assertTrue(receiver.isGenerationActive(newGeneration));
        receiver.stop();
        assertTrue(newSocket.closed.await(2L, TimeUnit.SECONDS));
    }

    @Test
    public void terminalTransportErrorTokenStaysCurrentUntilStopOrRestart() throws Exception {
        CountDownLatch reported = new CountDownLatch(1);
        AtomicLong callbackToken = new AtomicLong(-1L);
        UdpDecisionReceiver.Listener listener = new UdpDecisionReceiver.Listener() {
            @Override
            public void onPacket(
                    long generationToken, DecisionSnapshot packet, long receivedElapsedMs) {
                throw new AssertionError("unexpected packet");
            }

            @Override
            public void onMalformedPacket(long generationToken, String message) {
                throw new AssertionError(message);
            }

            @Override
            public void onTransportError(long generationToken, String message) {
                callbackToken.set(generationToken);
                reported.countDown();
            }
        };
        UdpDecisionReceiver receiver = new UdpDecisionReceiver(
                48_100,
                new DecisionPacketParser(),
                listener,
                () -> { throw new SocketException("test bind failure"); },
                () -> 123L);

        long generation = receiver.start();
        assertTrue(reported.await(2L, TimeUnit.SECONDS));
        assertEquals(generation, callbackToken.get());
        assertTrue(receiver.isGenerationActive(generation));
        receiver.stop();
        assertFalse(receiver.isGenerationActive(generation));
    }

    private static void awaitCount(CountDownLatch latch, long expectedRemaining)
            throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2L);
        while (latch.getCount() > expectedRemaining && System.nanoTime() < deadline) {
            Thread.sleep(5L);
        }
        assertEquals(expectedRemaining, latch.getCount());
    }

    private static final class NoOpListener implements UdpDecisionReceiver.Listener {
        @Override
        public void onPacket(
                long generationToken, DecisionSnapshot packet, long receivedElapsedMs) {}

        @Override
        public void onMalformedPacket(long generationToken, String message) {}

        @Override
        public void onTransportError(long generationToken, String message) {}
    }

    private static final class ControlledSocket extends DatagramSocket {
        final CountDownLatch bound = new CountDownLatch(1);
        final CountDownLatch closed = new CountDownLatch(1);
        private final byte[] payload;
        private boolean delivered;
        private boolean stopped;

        ControlledSocket(String payload) throws SocketException {
            super((SocketAddress) null);
            this.payload = payload == null
                    ? null : payload.getBytes(StandardCharsets.UTF_8);
        }

        @Override
        public synchronized void setReuseAddress(boolean on) {}

        @Override
        public synchronized void bind(SocketAddress address) {
            bound.countDown();
        }

        @Override
        public synchronized void setSoTimeout(int timeout) {}

        @Override
        public synchronized void receive(DatagramPacket packet) throws IOException {
            if (!delivered && payload != null) {
                System.arraycopy(payload, 0, packet.getData(), packet.getOffset(), payload.length);
                packet.setLength(payload.length);
                delivered = true;
                return;
            }
            while (!stopped) {
                try {
                    wait();
                } catch (InterruptedException error) {
                    Thread.currentThread().interrupt();
                    throw new SocketException("receive interrupted");
                }
            }
            throw new SocketException("socket closed");
        }

        @Override
        public synchronized void close() {
            if (!stopped) {
                stopped = true;
                closed.countDown();
                notifyAll();
            }
            super.close();
        }
    }
}
