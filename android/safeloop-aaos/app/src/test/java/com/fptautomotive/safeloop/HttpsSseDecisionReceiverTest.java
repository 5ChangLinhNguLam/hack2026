package com.fptautomotive.safeloop;

import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

public final class HttpsSseDecisionReceiverTest {
    private static final String TOKEN = "a234567890123456789012345678901";

    @Test
    public void parsesOnlyDataAndIgnoresHeartbeat() throws Exception {
        CountDownLatch packet = new CountDownLatch(1);
        RecordingListener listener = new RecordingListener(packet);
        String stream = ": heartbeat\n\ndata: " + DecisionFixtures.validJson() + "\n\n";
        HttpsSseDecisionReceiver receiver = receiver(
                new FixedConnection(200, "text/event-stream; charset=utf-8", stream),
                listener,
                delay -> Thread.sleep(10_000L));
        long generation = receiver.start();
        assertTrue(packet.await(2, TimeUnit.SECONDS));
        assertEquals(1, listener.packets.get());
        assertEquals(77L, listener.receivedMs);
        assertTrue(receiver.isGenerationActive(generation));
        receiver.stop();
        assertFalse(receiver.isGenerationActive(generation));
    }

    @Test
    public void rejectsRedirectAndWrongContentType() throws Exception {
        for (FixedConnection connection : new FixedConnection[] {
                new FixedConnection(302, "text/event-stream", ""),
                new FixedConnection(200, "application/json", "")}) {
            CountDownLatch error = new CountDownLatch(1);
            RecordingListener listener = new RecordingListener(new CountDownLatch(1));
            listener.transportError = error;
            HttpsSseDecisionReceiver receiver = receiver(
                    connection, listener, delay -> { throw new InterruptedException(); });
            receiver.start();
            assertTrue(error.await(2, TimeUnit.SECONDS));
            receiver.stop();
            assertEquals(0, listener.packets.get());
        }
    }

    @Test
    public void oversizeLineIsRejectedBeforeJsonAllocation() throws Exception {
        byte[] input = new byte[HttpsSseDecisionReceiver.MAX_SSE_LINE_BYTES + 2];
        for (int index = 0; index < input.length; index++) input[index] = 'x';
        try {
            HttpsSseDecisionReceiver.readLine(new ByteArrayInputStream(input));
            fail("oversize SSE line accepted");
        } catch (IOException expected) {
            assertTrue(expected.getMessage().contains("maximum"));
        }
    }

    @Test
    public void reconnectUsesBoundedExponentialSchedule() throws Exception {
        AtomicInteger opens = new AtomicInteger();
        List<Long> delays = new ArrayList<>();
        CountDownLatch three = new CountDownLatch(3);
        RecordingListener listener = new RecordingListener(new CountDownLatch(1));
        HttpsSseDecisionReceiver receiver = new HttpsSseDecisionReceiver(
                new URL("https://safeloop.sonnet.io.vn/v1/decisions/stream"), TOKEN,
                new DecisionPacketParser(), listener,
                (url, token) -> {
                    opens.incrementAndGet();
                    return new FixedConnection(200, "text/event-stream", "");
                }, () -> 77L,
                delay -> {
                    synchronized (delays) { delays.add(delay); }
                    three.countDown();
                    if (three.getCount() > 0) Thread.sleep(1L);
                    else Thread.sleep(10_000L);
                });
        receiver.start();
        assertTrue(three.await(2, TimeUnit.SECONDS));
        receiver.stop();
        assertTrue(opens.get() >= 3);
        synchronized (delays) {
            assertTrue(delays.get(0) >= 1_000L && delays.get(0) <= 1_250L);
            assertTrue(delays.get(1) >= 2_000L && delays.get(1) <= 2_250L);
            assertTrue(delays.get(2) >= 4_000L && delays.get(2) <= 4_250L);
        }
    }

    @Test
    public void urlMustBeHttpsAndMustNotCarryToken() {
        for (String value : new String[] {
                "http://safeloop.sonnet.io.vn/v1/decisions/stream",
                "https://token@safeloop.sonnet.io.vn/v1/decisions/stream",
                "https://safeloop.sonnet.io.vn/v1/decisions/stream?token=x"}) {
            try {
                HttpsSseDecisionReceiver.validateUrl(value);
                fail("unsafe stream URL accepted");
            } catch (IllegalArgumentException expected) {
                // Expected.
            }
        }
    }

    private static HttpsSseDecisionReceiver receiver(
            HttpsSseDecisionReceiver.Connection connection,
            RecordingListener listener,
            HttpsSseDecisionReceiver.Sleeper sleeper) throws Exception {
        return new HttpsSseDecisionReceiver(
                new URL("https://safeloop.sonnet.io.vn/v1/decisions/stream"), TOKEN,
                new DecisionPacketParser(), listener, (url, token) -> connection,
                () -> 77L, sleeper);
    }

    private static final class FixedConnection implements HttpsSseDecisionReceiver.Connection {
        private final int status;
        private final String contentType;
        private final byte[] body;
        FixedConnection(int status, String contentType, String body) {
            this.status = status;
            this.contentType = contentType;
            this.body = body.getBytes(StandardCharsets.UTF_8);
        }
        @Override public int responseCode() { return status; }
        @Override public String contentType() { return contentType; }
        @Override public InputStream inputStream() { return new ByteArrayInputStream(body); }
        @Override public void disconnect() {}
    }

    private static final class RecordingListener implements DecisionReceiver.Listener {
        final AtomicInteger packets = new AtomicInteger();
        final CountDownLatch packet;
        volatile CountDownLatch transportError = new CountDownLatch(1);
        volatile long receivedMs;
        RecordingListener(CountDownLatch packet) { this.packet = packet; }
        @Override public void onPacket(long token, DecisionSnapshot value, long received) {
            receivedMs = received;
            packets.incrementAndGet();
            packet.countDown();
        }
        @Override public void onMalformedPacket(long token, String message) {}
        @Override public void onTransportError(long token, String message) {
            transportError.countDown();
        }
    }
}
