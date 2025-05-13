package org.brain4j.core.loss;

import org.brain4j.math.tensor.Tensor;

/**
 * Interface defining a org.brain4j.core.loss function
 */
public interface LossFunction {
    /**
     * Calculates the error of the network given the expected and predicted tensor.
     * @param expected The expected output
     * @param predicted The predicted output
     * @return A scalar value
     */
    double calculate(Tensor expected, Tensor predicted);

    /**
     * Calculates the delta for the last org.brain4j.core.layer of the network.
     * @param error Calculated as <code>expected - predicted</code>
     * @param derivative The derivative of the activation function
     * @return The delta tensor
     */
    Tensor getDelta(Tensor error, Tensor derivative);
}
