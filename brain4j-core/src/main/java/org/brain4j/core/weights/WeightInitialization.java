package org.brain4j.core.weights;

/**
 * Strategies for initializing neural network weights.
 * Each weight is initialized randomly within bounds determined by the
 * chosen initialization technique and the number of input and output neurons
 */
public interface WeightInitialization {
    /**
     * Gets the maximum bound of each weight when initialized.
     * @param input the amount of input neurons
     * @param output the amount of output neurons
     * @return a scalar value
     */
    double getBound(int input, int output);
}
