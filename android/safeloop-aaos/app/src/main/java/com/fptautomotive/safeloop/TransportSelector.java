package com.fptautomotive.safeloop;

/** Pure selection rule: a cloud URL suppresses UDP for the complete Activity run. */
final class TransportSelector {
    enum Transport { UDP, HTTPS_SSE }

    private TransportSelector() {}

    static Transport select(String cloudStreamUrl) {
        return cloudStreamUrl == null || cloudStreamUrl.trim().isEmpty()
                ? Transport.UDP : Transport.HTTPS_SSE;
    }
}
