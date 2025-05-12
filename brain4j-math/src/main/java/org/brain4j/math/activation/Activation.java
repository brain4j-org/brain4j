package org.brain4j.math.activation;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

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
        int[] shape = input.shape();
        Tensor result = Tensors.create(shape);

        if (input.dimension() == 2) {
            for (int i = 0; i < shape[0]; i++) {
                for (int j = 0; j < shape[1]; j++) {
                    result.set(activate(input.get(i, j)), i, j);
                }
            }
        } else {
            for (int i = 0; i < input.elements(); i++) {
                result.set(activate(input.get(i)), i);
            }
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
        int[] shape = input.shape();
        Tensor result = Tensors.create(shape);

        if (input.dimension() == 2) {
            for (int i = 0; i < shape[0]; i++) {
                for (int j = 0; j < shape[1]; j++) {
                    result.set(getDerivative(input.get(i, j)), i, j);
                }
            }
        } else {
            for (int i = 0; i < input.elements(); i++) {
                result.set(getDerivative(input.get(i)), i);
            }
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
