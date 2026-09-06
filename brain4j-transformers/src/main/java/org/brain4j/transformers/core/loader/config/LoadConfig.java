package org.brain4j.transformers.core.loader.config;

public record LoadConfig(String split, boolean streaming, boolean forceDownload, int maxFiles) {

    public static LoadConfig defaultConfig() {
        return new LoadConfig("all", false, false, Integer.MAX_VALUE);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String split = "all";
        private boolean streaming = false;
        private boolean forceDownload = false;
        private int maxFiles = Integer.MAX_VALUE;

        public Builder setSplit(String split) {
            this.split = split;
            return this;
        }

        public Builder setStreaming(boolean streaming) {
            this.streaming = streaming;
            return this;
        }

        public Builder setForceDownload(boolean forceDownload) {
            this.forceDownload = forceDownload;
            return this;
        }

        public Builder setMaxFiles(int maxFiles) {
            this.maxFiles = maxFiles;
            return this;
        }

        public LoadConfig build() {
            return new LoadConfig(split, streaming, forceDownload, maxFiles);
        }
    }
}