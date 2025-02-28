package net.echo.brain4j.loss;

import net.echo.brain4j.utils.Vector;

/**
 * Also known as cost function is used to evaluate the model's performance while training.
 */
public interface LossFunction {

    /**
     * Compares the predicted vector from the actual vector.
     *
     * @param actual    the vector predicted by the model
     * @param predicted the vector we should expect
     *
     * @return a number that describes the model's loss
     */
    double calculate(Vector actual, Vector predicted);

    /**
     * Computes the delta for the given error and derivative.
     *
     * @param error the error between the predicted and actual values
     * @param derivative the derivative of the activation function
     *
     * @return the delta value
     */
    double getDelta(double error, double derivative);
}
