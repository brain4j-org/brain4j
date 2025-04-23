package net.echo.brain4j.activation;

import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.Tensors;

public interface Activation {

    /**
     * Activate a single scalar value (e.g. ReLU, Sigmoid, Tanh, etc.).
     */
    double activate(double input);

    /**
     * Activate a vector of values.
     * Return a new vector containing the activated values.
     */
    default Tensor activate(Tensor input) {
        if (input.dimension() > 1) {
            throw new IllegalArgumentException("Activation only supports 1D tensors!");
        }

        Tensor result = Tensors.create(input.elements());

        for (int i = 0; i < input.elements(); i++) {
            double value = activate(input.get(i));
            result.set(value, i);
        }

        return result;
    }

    /**
     * Get the derivative (scalar) of the activation at a single value.
     */
    double getDerivative(double input);

    /**
     * Get the derivative (vector) of the activation at a vector of values.
     */
    default Tensor getDerivative(Tensor input) {
        if (input.dimension() > 1) {
            throw new IllegalArgumentException("Derivative only supports 1D tensors!");
        }

        Tensor result = Tensors.create(input.elements());

        for (int i = 0; i < input.elements(); i++) {
            double value = getDerivative(input.get(i));
            result.set(value, i);
        }

        return result;
    }

    /**
     * Gets the default name for this activation function.
     * @return The name of the activation function.
     */
    default String getName() {
        return getClass().getSimpleName().replaceAll("Activation", "");
    }
}
