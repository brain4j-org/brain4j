package org.brain4j.transformers.core.model;

import java.util.Random;

/**
 * Configuration object defining the sampling strategy used during
 * token generation in a Language Model (LM).
 * <p>
 * This configuration controls both the stochasticity of the generation
 * process and the constraints applied to the output sequence.
 *
 * @param random
 *        Source of randomness used for sampling tokens. Providing a
 *        deterministic {@link Random} instance allows reproducible generations.
 *
 * @param maxLength
 *        Maximum number of tokens that may be generated. A negative value
 *        indicates that no explicit length limit is enforced by this configuration.
 *
 * @param topK
 *        Limits sampling to the {@code K} most probable tokens after softmax.
 *        Only the top {@code K} tokens with the highest probabilities are
 *        considered, and one of them is sampled according to their normalized
 *        distribution. Typical values range from 5 to 100.
 *
 * @param temperature
 *        Temperature applied to the logits before the softmax operation.
 *        Values {@code < 1.0} reduce randomness and make the output more
 *        deterministic, while values {@code > 1.0} increase diversity.
 *        A value of {@code 1.0} leaves the distribution unchanged.
 */
public record SamplingConfig(Random random, int maxLength, int topK, double temperature) {

    /**
     * Returns a default sampling configuration suitable for general-purpose
     * text generation.
     *
     * @return a default {@link SamplingConfig} instance
     */
    public static SamplingConfig defaultConfig() {
        return new SamplingConfig(new Random(), 32, 5, 1.0);
    }

    /**
     * Creates a new {@link Builder} for constructing a {@link SamplingConfig}
     * with custom parameters.
     *
     * @return a new {@link Builder} instance
     */
    public static SamplingConfig.Builder builder() {
        return new SamplingConfig.Builder();
    }

    /**
     * Builder for {@link SamplingConfig}, allowing selective customization
     * of sampling parameters before construction.
     */
    public static class Builder {

        private Random random = new Random();
        private int maxLength = -1;
        private int topK = 50;
        private double temperature = 1.0;

        /**
         * Sets the random number generator used for token sampling.
         *
         * @param random the {@link Random} instance to use
         * @return this builder instance
         */
        public Builder setRandom(Random random) {
            this.random = random;
            return this;
        }

        /**
         * Sets the maximum number of tokens to generate.
         *
         * @param maxLength the maximum generation length, or a negative value
         *                  to disable length limiting
         * @return this builder instance
         */
        public Builder maxLength(int maxLength) {
            this.maxLength = maxLength;
            return this;
        }

        /**
         * Sets the {@code top-k} value used to restrict the sampling space.
         *
         * @param topK number of highest-probability tokens to consider
         * @return this builder instance
         */
        public Builder setTopK(int topK) {
            this.topK = topK;
            return this;
        }

        /**
         * Sets the temperature used during softmax scaling.
         *
         * @param temperature temperature value; must be strictly positive
         * @return this builder instance
         */
        public Builder setTemperature(double temperature) {
            this.temperature = temperature;
            return this;
        }

        /**
         * Builds the {@link SamplingConfig} instance.
         *
         * @return a new {@link SamplingConfig}
         */
        public SamplingConfig build() {
            return new SamplingConfig(random, maxLength, topK, temperature);
        }
    }
}