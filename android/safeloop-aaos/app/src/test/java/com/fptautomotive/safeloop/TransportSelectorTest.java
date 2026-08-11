package com.fptautomotive.safeloop;

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public final class TransportSelectorTest {
    @Test
    public void cloudExtraAndUdpAreMutuallyExclusive() {
        assertEquals(TransportSelector.Transport.UDP, TransportSelector.select(null));
        assertEquals(TransportSelector.Transport.UDP, TransportSelector.select("  "));
        assertEquals(
                TransportSelector.Transport.HTTPS_SSE,
                TransportSelector.select(
                        "https://safeloop.sonnet.io.vn/v1/decisions/stream"));
    }
}
