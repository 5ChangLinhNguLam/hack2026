package com.fptautomotive.safeloop;

import android.content.Context;
import android.net.ConnectivityManager;
import android.net.Network;
import android.net.NetworkCapabilities;
import android.net.NetworkRequest;

import java.io.IOException;
import java.net.URL;
import java.net.URLConnection;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Opens each request through a fresh, explicitly selected Android Network. */
final class AndroidNetworkConnectionOpener implements AutoCloseable {
    private final ConnectivityManager connectivity;
    private final Object lock = new Object();
    private final Map<Network, NetworkCapabilities> available = new LinkedHashMap<>();
    private final ConnectivityManager.NetworkCallback callback;
    private boolean closed;

    AndroidNetworkConnectionOpener(Context context) {
        if (context == null) {
            throw new IllegalArgumentException("Android context is required");
        }
        Context application = context.getApplicationContext();
        Context serviceContext = application == null ? context : application;
        connectivity = (ConnectivityManager) serviceContext.getSystemService(
                Context.CONNECTIVITY_SERVICE);
        if (connectivity == null) {
            throw new IllegalStateException("ConnectivityManager is unavailable");
        }
        callback = new ConnectivityManager.NetworkCallback() {
            @Override
            public void onAvailable(Network network) {
                update(network, connectivity.getNetworkCapabilities(network));
            }

            @Override
            public void onCapabilitiesChanged(
                    Network network, NetworkCapabilities capabilities) {
                update(network, capabilities);
            }

            @Override
            public void onLost(Network network) {
                synchronized (lock) {
                    available.remove(network);
                }
            }
        };
        NetworkRequest request = new NetworkRequest.Builder()
                .addCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
                .build();
        connectivity.registerNetworkCallback(request, callback);
        Network active = connectivity.getActiveNetwork();
        if (active != null) {
            update(active, connectivity.getNetworkCapabilities(active));
        }
    }

    URLConnection open(URL url) throws IOException {
        if (url == null) {
            throw new IllegalArgumentException("URL is required");
        }
        try {
            Network network = InternetNetworkSelector.select(snapshot());
            // Network.openConnection binds both DNS and the socket to this route.
            return network.openConnection(url);
        } catch (SecurityException error) {
            throw new IOException("Android network state is unavailable", error);
        }
    }

    private void update(Network network, NetworkCapabilities capabilities) {
        if (network == null) {
            return;
        }
        synchronized (lock) {
            if (!closed) {
                if (capabilities == null) {
                    available.remove(network);
                } else {
                    available.put(network, capabilities);
                }
            }
        }
    }

    private List<InternetNetworkSelector.Candidate<Network>> snapshot()
            throws IOException {
        List<InternetNetworkSelector.Candidate<Network>> candidates = new ArrayList<>();
        Network active = connectivity.getActiveNetwork();
        if (active != null) {
            NetworkCapabilities capabilities = connectivity.getNetworkCapabilities(active);
            if (capabilities != null) {
                candidates.add(new AndroidCandidate(active, true, capabilities));
            }
        }
        synchronized (lock) {
            if (closed) {
                throw new IOException("Android network selector is closed");
            }
            for (Map.Entry<Network, NetworkCapabilities> entry : available.entrySet()) {
                if (!entry.getKey().equals(active)) {
                    candidates.add(new AndroidCandidate(
                            entry.getKey(), false, entry.getValue()));
                }
            }
        }
        return candidates;
    }

    @Override
    public void close() {
        synchronized (lock) {
            if (closed) {
                return;
            }
            closed = true;
            available.clear();
        }
        connectivity.unregisterNetworkCallback(callback);
    }

    private static final class AndroidCandidate
            implements InternetNetworkSelector.Candidate<Network> {
        private final Network network;
        private final boolean active;
        private final NetworkCapabilities capabilities;

        AndroidCandidate(
                Network network, boolean active, NetworkCapabilities capabilities) {
            this.network = network;
            this.active = active;
            this.capabilities = capabilities;
        }

        @Override public Network network() { return network; }
        @Override public boolean isActive() { return active; }
        @Override public boolean isWifi() {
            return capabilities != null
                    && capabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI);
        }
        @Override public boolean hasInternet() {
            return capabilities != null
                    && capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET);
        }
        @Override public boolean isValidated() {
            return capabilities != null
                    && capabilities.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED);
        }
        @Override public boolean isNotSuspended() {
            return capabilities != null
                    && capabilities.hasCapability(
                            NetworkCapabilities.NET_CAPABILITY_NOT_SUSPENDED);
        }
    }
}
