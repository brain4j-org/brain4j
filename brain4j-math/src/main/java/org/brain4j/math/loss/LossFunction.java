package org.brain4j.math.loss;

import org.brain4j.math.loss.impl.BinaryCrossEntropy;
import org.brain4j.math.loss.impl.CrossEntropy;
import org.brain4j.math.loss.impl.MeanAbsoluteError;
import org.brain4j.math.loss.impl.MeanSquaredError;
import org.brain4j.math.tensor.Tensor;

/**
 * Loss functions (also called cost functions) are used during training
 * for backpropagation and to measure the performance of a network.
 *
 * @author xEcho1337
 * @see CrossEntropy
 * @see BinaryCrossEntropy
 * @see MeanAbsoluteError
 * @see MeanSquaredError
 */
public interface LossFunction {
    /**
     * Calculates the error of the network given the expected and predicted tensor.
     * @param expected the expected output
     * @param predicted the predicted output
     * @return a scalar value
     */
    double calculate(Tensor expected, Tensor predicted);

    /**
     * Calculates the delta for the last org.brain4j.core.layer of the network.
     * @param output the output produced by the model
     * @param target the expected output
     * @param derivative the derivative of the activation function
     * @return the delta tensor
     */
    Tensor delta(Tensor output, Tensor target, Tensor derivative);

    /**
     * Gets whether this loss function is typically used for regression.
     * @return true if this is used for regression, false otherwise
     */
    boolean isRegression();
}
