package com.fptautomotive.safeloop;

import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

public final class InternetNetworkSelectorTest {
    @Test
    public void prefersValidatedWifiOverActiveValidatedEthernet() throws Exception {
        String selected = InternetNetworkSelector.select(Arrays.asList(
                candidate("ethernet", true, false, true, true, true),
                candidate("wifi", false, true, true, true, true)));
        assertEquals("wifi", selected);
    }

    @Test
    public void prefersValidatedInternetOverUnvalidatedWifi() throws Exception {
        String selected = InternetNetworkSelector.select(Arrays.asList(
                candidate("wifi", false, true, true, false, true),
                candidate("cellular", true, false, true, true, true)));
        assertEquals("cellular", selected);
    }

    @Test
    public void fallsBackToUnvalidatedWifiAndThenActiveInternet() throws Exception {
        String wifi = InternetNetworkSelector.select(Arrays.asList(
                candidate("ethernet", true, false, true, false, true),
                candidate("wifi", false, true, true, false, true)));
        assertEquals("wifi", wifi);

        String active = InternetNetworkSelector.select(Arrays.asList(
                candidate("secondary", false, false, true, false, true),
                candidate("active", true, false, true, false, true)));
        assertEquals("active", active);
    }

    @Test
    public void rejectsSuspendedOrNonInternetNetworks() {
        try {
            InternetNetworkSelector.select(Arrays.asList(
                    candidate("suspended-wifi", true, true, true, true, false),
                    candidate("local-only", false, true, false, true, true)));
            fail("unusable network was selected");
        } catch (IOException expected) {
            assertEquals("no usable Internet network is available", expected.getMessage());
        }

        try {
            InternetNetworkSelector.select(Collections.emptyList());
            fail("empty network list was accepted");
        } catch (IOException expected) {
            assertEquals("no usable Internet network is available", expected.getMessage());
        }
    }

    private static InternetNetworkSelector.Candidate<String> candidate(
            String name,
            boolean active,
            boolean wifi,
            boolean internet,
            boolean validated,
            boolean notSuspended) {
        return new FakeCandidate(
                name, active, wifi, internet, validated, notSuspended);
    }

    private static final class FakeCandidate
            implements InternetNetworkSelector.Candidate<String> {
        private final String name;
        private final boolean active;
        private final boolean wifi;
        private final boolean internet;
        private final boolean validated;
        private final boolean notSuspended;

        FakeCandidate(
                String name,
                boolean active,
                boolean wifi,
                boolean internet,
                boolean validated,
                boolean notSuspended) {
            this.name = name;
            this.active = active;
            this.wifi = wifi;
            this.internet = internet;
            this.validated = validated;
            this.notSuspended = notSuspended;
        }

        @Override public String network() { return name; }
        @Override public boolean isActive() { return active; }
        @Override public boolean isWifi() { return wifi; }
        @Override public boolean hasInternet() { return internet; }
        @Override public boolean isValidated() { return validated; }
        @Override public boolean isNotSuspended() { return notSuspended; }
    }
}
