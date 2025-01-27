package net.echo.brain4j.loss;

import net.echo.brain4j.loss.impl.*;

/**
 * Enum representing common loss functions used in neural networks.
 * Each constant maps to a specific implementation of {@link LossFunction}.
 */
public enum LossFunctions {

    /**
     * Mean Squared Error (MSE):
     * Calculates the average of the squared differences between predicted and actual values.
     * Commonly used for regression problems.
     */
    MEAN_SQUARED_ERROR(new MeanSquaredError()),

    /**
     * Mean Absolute Error (MAE):
     * Computes the mean of the absolute differences between predicted and actual values.
     * Robust to outliers and used in regression tasks.
     */
    MEAN_ABSOLUTE_ERROR(new MeanAbsoluteError()),

    /**
     * Binary Cross-Entropy:
     * Measures the performance of a classification model with binary outcomes.
     * Compares predicted probabilities to true binary labels.
     */
    BINARY_CROSS_ENTROPY(new BinaryCrossEntropy()),

    /**
     * Cross-Entropy:
     * Used for multi-class classification problems.
     * Evaluates the divergence between predicted probability distributions and true labels.
     */
    CROSS_ENTROPY(new CrossEntropy());

    private final LossFunction function;

    /**
     * Constructor that associates a specific loss function implementation.
     *
     * @param function The implementation of the loss function.
     */
    LossFunctions(LossFunction function) {
        this.function = function;
    }

    /**
     * Retrieves the loss function implementation.
     *
     * @return The associated {@link LossFunction}.
     */
    public LossFunction getFunction() {
        return function;
    }
}
