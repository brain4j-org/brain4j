package net.echo.brain4j.loss;

import net.echo.brain4j.loss.impl.BinaryCrossEntropy;
import net.echo.brain4j.loss.impl.CrossEntropy;
import net.echo.brain4j.loss.impl.MeanAbsoluteError;
import net.echo.brain4j.loss.impl.MeanSquaredError;

/**
 * Also known as cost function, measures the model's performance and is minimized during training.
 * Each element delegates to a specific implementation of {@link LossFunction}.
 */
public enum Loss {

    /**
     * Mean Squared Error (MSE): Calculates the average of the squared differences between predicted and actual values.
     * It's commonly used in regression problems.
     */
    MEAN_SQUARED_ERROR(new MeanSquaredError()),

    /**
     * Mean Absolute Error (MAE): Computes the mean of the absolute differences between predicted and actual values.
     * Robust to outliers and used in regression tasks.
     */
    MEAN_ABSOLUTE_ERROR(new MeanAbsoluteError()),

    /**
     * Binary Cross-Entropy: Measures the performance of a classification model with binary outcomes. Compares predicted
     * probabilities to the true binary labels.
     */
    BINARY_CROSS_ENTROPY(new BinaryCrossEntropy()),

    /**
     * Cross-Entropy: Used in categorical classification problems. Evaluates the divergence between predicted probability
     * distributions and true labels.
     */
    CROSS_ENTROPY(new CrossEntropy());

    private final LossFunction function;

    /**
     * Constructor that associates a specific loss function implementation.
     * @param function The loss function implementation.
     */
    Loss(LossFunction function) {
        this.function = function;
    }

    /**
     * Gets the associated implementation of this loss function.
     * @return The associated {@link LossFunction}.
     */
    public LossFunction getFunction() {
        return function;
    }
}
