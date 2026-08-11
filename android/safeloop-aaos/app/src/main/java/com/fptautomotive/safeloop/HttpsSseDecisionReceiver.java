package com.fptautomotive.safeloop;

import android.os.SystemClock;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.concurrent.ThreadLocalRandom;

import javax.net.ssl.HttpsURLConnection;

/** Outbound-only HTTPS SSE receiver for the AWS fast-demo decision stream. */
public final class HttpsSseDecisionReceiver implements DecisionReceiver {
    static final int CONNECT_TIMEOUT_MS = 5_000;
    static final int READ_TIMEOUT_MS = 15_000;
    static final int MAX_SSE_LINE_BYTES = DecisionPacketParser.MAX_PACKET_BYTES + 6;
    private static final long[] BACKOFF_MS = {1_000L, 2_000L, 4_000L, 8_000L, 10_000L};

    interface Connection {
        int responseCode() throws IOException;
        String contentType();
        InputStream inputStream() throws IOException;
        void disconnect();
    }

    interface ConnectionFactory {
        Connection open(URL url, String bearerToken) throws IOException;
    }

    interface Sleeper {
        void sleep(long delayMs) throws InterruptedException;
    }

    interface ElapsedClock {
        long nowMs();
    }

    private static final class PlatformConnection implements Connection {
        private final HttpsURLConnection connection;

        PlatformConnection(URL url, String token) throws IOException {
            HttpURLConnection candidate = (HttpURLConnection) url.openConnection();
            if (!(candidate instanceof HttpsURLConnection)) {
                candidate.disconnect();
                throw new IOException("decision stream must use HTTPS");
            }
            connection = (HttpsURLConnection) candidate;
            connection.setInstanceFollowRedirects(false);
            connection.setRequestMethod("GET");
            connection.setConnectTimeout(CONNECT_TIMEOUT_MS);
            connection.setReadTimeout(READ_TIMEOUT_MS);
            connection.setUseCaches(false);
            connection.setRequestProperty("Accept", "text/event-stream");
            connection.setRequestProperty("Cache-Control", "no-cache");
            connection.setRequestProperty("Authorization", "Bearer " + token);
        }

        @Override public int responseCode() throws IOException { return connection.getResponseCode(); }
        @Override public String contentType() { return connection.getContentType(); }
        @Override public InputStream inputStream() throws IOException { return connection.getInputStream(); }
        @Override public void disconnect() { connection.disconnect(); }
    }

    private final URL streamUrl;
    private final String bearerToken;
    private final DecisionPacketParser parser;
    private final DecisionReceiver.Listener listener;
    private final ConnectionFactory connectionFactory;
    private final ElapsedClock elapsedClock;
    private final Sleeper sleeper;
    private boolean running;
    private long generation;
    private Thread thread;
    private Connection connection;

    public HttpsSseDecisionReceiver(
            String streamUrl,
            String bearerToken,
            DecisionPacketParser parser,
            DecisionReceiver.Listener listener) {
        this(
                validateUrl(streamUrl),
                validateToken(bearerToken),
                parser,
                listener,
                PlatformConnection::new,
                SystemClock::elapsedRealtime,
                Thread::sleep);
    }

    HttpsSseDecisionReceiver(
            URL streamUrl,
            String bearerToken,
            DecisionPacketParser parser,
            DecisionReceiver.Listener listener,
            ConnectionFactory connectionFactory,
            ElapsedClock elapsedClock,
            Sleeper sleeper) {
        if (streamUrl == null || parser == null || listener == null
                || connectionFactory == null || elapsedClock == null || sleeper == null) {
            throw new IllegalArgumentException("SSE receiver dependencies are required");
        }
        this.streamUrl = streamUrl;
        this.bearerToken = validateToken(bearerToken);
        this.parser = parser;
        this.listener = listener;
        this.connectionFactory = connectionFactory;
        this.elapsedClock = elapsedClock;
        this.sleeper = sleeper;
    }

    private static URL validateUrl(String value) {
        try {
            URL url = new URL(value == null ? "" : value);
            if (!"https".equalsIgnoreCase(url.getProtocol())
                    || url.getUserInfo() != null
                    || url.getQuery() != null
                    || url.getRef() != null) {
                throw new IllegalArgumentException(
                        "cloud_stream_url must be credential-free HTTPS");
            }
            return url;
        } catch (IOException error) {
            throw new IllegalArgumentException("cloud_stream_url is invalid", error);
        }
    }

    private static String validateToken(String value) {
        if (value == null || value.length() < 24 || value.indexOf('\n') >= 0
                || value.indexOf('\r') >= 0) {
            throw new IllegalArgumentException("cloud stream bearer token is invalid");
        }
        return value;
    }

    @Override
    public synchronized long start() {
        if (running) {
            return generation;
        }
        running = true;
        long activeGeneration = ++generation;
        thread = new Thread(() -> receiveLoop(activeGeneration), "safeloop-https-sse");
        thread.setDaemon(true);
        thread.start();
        return activeGeneration;
    }

    private void receiveLoop(long activeGeneration) {
        int backoffIndex = 0;
        while (isActive(activeGeneration)) {
            Connection active = null;
            try {
                active = connectionFactory.open(streamUrl, bearerToken);
                if (!claimConnection(activeGeneration, active)) {
                    active.disconnect();
                    return;
                }
                int responseCode = active.responseCode();
                if (responseCode != HttpURLConnection.HTTP_OK) {
                    throw new IOException("SSE rejected HTTP status " + responseCode);
                }
                String contentType = active.contentType();
                if (contentType == null || !"text/event-stream".equals(
                        contentType.split(";", 2)[0].trim().toLowerCase(Locale.US))) {
                    throw new IOException("SSE response has invalid Content-Type");
                }
                if (readEvents(activeGeneration, active.inputStream())) {
                    backoffIndex = 0;
                }
                if (isActive(activeGeneration)) {
                    throw new IOException("SSE stream ended");
                }
            } catch (DecisionPacketParser.PacketFormatException error) {
                reportMalformed(activeGeneration, error.getMessage());
            } catch (CharacterCodingException error) {
                reportMalformed(activeGeneration, "SSE event is not valid UTF-8");
            } catch (IOException error) {
                reportTransportError(activeGeneration, safeMessage(error));
            } finally {
                clearConnection(active);
                if (active != null) {
                    active.disconnect();
                }
            }
            if (!isActive(activeGeneration)) {
                return;
            }
            long base = BACKOFF_MS[Math.min(backoffIndex, BACKOFF_MS.length - 1)];
            backoffIndex = Math.min(backoffIndex + 1, BACKOFF_MS.length - 1);
            long jitter = ThreadLocalRandom.current().nextLong(0L, 251L);
            try {
                sleeper.sleep(base + jitter);
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            }
        }
    }

    private boolean readEvents(long activeGeneration, InputStream input)
            throws IOException, DecisionPacketParser.PacketFormatException {
        boolean receivedPacket = false;
        while (isActive(activeGeneration)) {
            byte[] line = readLine(input);
            if (line == null) {
                return receivedPacket;
            }
            if (line.length == 0 || line[0] == ':') {
                continue;
            }
            String decoded = decodeUtf8(line);
            if (!decoded.startsWith("data: ")) {
                continue;
            }
            String payload = decoded.substring(6);
            if (payload.getBytes(StandardCharsets.UTF_8).length
                    > DecisionPacketParser.MAX_PACKET_BYTES) {
                throw new DecisionPacketParser.PacketFormatException(
                        "SSE decision exceeds maximum size");
            }
            DecisionSnapshot packet = parser.parse(payload);
            reportPacket(activeGeneration, packet, elapsedClock.nowMs());
            receivedPacket = true;
        }
        return receivedPacket;
    }

    static byte[] readLine(InputStream input) throws IOException {
        ByteArrayOutputStream line = new ByteArrayOutputStream(256);
        while (true) {
            int value = input.read();
            if (value < 0) {
                return line.size() == 0 ? null : line.toByteArray();
            }
            if (value == '\n') {
                byte[] result = line.toByteArray();
                if (result.length > 0 && result[result.length - 1] == '\r') {
                    byte[] trimmed = new byte[result.length - 1];
                    System.arraycopy(result, 0, trimmed, 0, trimmed.length);
                    return trimmed;
                }
                return result;
            }
            if (line.size() >= MAX_SSE_LINE_BYTES) {
                throw new IOException("SSE line exceeds maximum size");
            }
            line.write(value);
        }
    }

    private static String decodeUtf8(byte[] value) throws CharacterCodingException {
        return StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(value))
                .toString();
    }

    private static String safeMessage(IOException error) {
        String message = error.getMessage();
        return message == null || message.isEmpty() ? "HTTPS SSE transport failed" : message;
    }

    private synchronized boolean claimConnection(long activeGeneration, Connection candidate) {
        if (!running || generation != activeGeneration) {
            return false;
        }
        connection = candidate;
        return true;
    }

    private synchronized void clearConnection(Connection candidate) {
        if (connection == candidate) {
            connection = null;
        }
    }

    @Override
    public synchronized boolean isGenerationActive(long token) {
        return running && token > 0L && generation == token;
    }

    private boolean isActive(long token) {
        return isGenerationActive(token);
    }

    private void reportPacket(long token, DecisionSnapshot packet, long receivedMs) {
        if (isGenerationActive(token)) listener.onPacket(token, packet, receivedMs);
    }

    private void reportMalformed(long token, String message) {
        if (isGenerationActive(token)) listener.onMalformedPacket(token, message);
    }

    private void reportTransportError(long token, String message) {
        if (isGenerationActive(token)) listener.onTransportError(token, message);
    }

    @Override
    public synchronized void stop() {
        running = false;
        generation++;
        Connection active = connection;
        connection = null;
        if (active != null) active.disconnect();
        Thread activeThread = thread;
        thread = null;
        if (activeThread != null) activeThread.interrupt();
    }

    @Override public void close() { stop(); }
}
