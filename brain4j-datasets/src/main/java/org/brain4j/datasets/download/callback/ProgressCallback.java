package org.brain4j.datasets.download.callback;

@FunctionalInterface
public interface ProgressCallback {

    void onProgress(String filename, double percentage, String message);

    /**
     * No-operation progress callback.
     */
    ProgressCallback NOOP = (filename, percentage, message) -> {};
}