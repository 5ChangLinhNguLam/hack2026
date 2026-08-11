package com.fptautomotive.safeloop;

import java.io.IOException;
import java.util.List;

/** Chooses one currently usable Internet route without binding the whole process. */
final class InternetNetworkSelector {
    interface Candidate<T> {
        T network();
        boolean isActive();
        boolean isWifi();
        boolean hasInternet();
        boolean isValidated();
        boolean isNotSuspended();
    }

    private InternetNetworkSelector() {}

    static <T> T select(List<? extends Candidate<T>> candidates) throws IOException {
        T selected = null;
        int selectedScore = Integer.MIN_VALUE;
        if (candidates != null) {
            for (Candidate<T> candidate : candidates) {
                if (candidate == null || candidate.network() == null
                        || !candidate.hasInternet() || !candidate.isNotSuspended()) {
                    continue;
                }
                int score = (candidate.isValidated() ? 100 : 0)
                        + (candidate.isWifi() ? 10 : 0)
                        + (candidate.isActive() ? 1 : 0);
                if (selected == null || score > selectedScore) {
                    selected = candidate.network();
                    selectedScore = score;
                }
            }
        }
        if (selected == null) {
            throw new IOException("no usable Internet network is available");
        }
        return selected;
    }
}
