package org.brain4j.core.weights;

/**
 * Interface defining a weight initialization technique.
 */
public interface WeightInitialization {
    /**
     * Gets the maximum bound of each weight when initialized.
     * @param input The amount of input neurons
     * @param output The amount of output neurons
     * @return A scalar value
     */
    double getBound(int input, int output);
}
