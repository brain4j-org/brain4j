package net.echo.brain4j.activation;

import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;
import net.echo.math4j.math.vector.Vector;

import java.util.List;

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

        Tensor result = TensorFactory.create(input.elements());

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

        Tensor result = TensorFactory.create(input.elements());

        for (int i = 0; i < input.elements(); i++) {
            double value = getDerivative(input.get(i));
            result.set(value, i);
        }

        return result;
    }

    /**
     * Applies the activation factor to a list of neurons.
     */
    default void apply(StatesCache cacheHolder, List<Neuron> neurons) {
        for (Neuron neuron : neurons) {
            double output = neuron.getValue(cacheHolder) + neuron.getBias();
            double activated = activate(output);

            neuron.setValue(cacheHolder, activated);
        }
    }

    /**
     * Gets the default name for this activation function.
     * @return The name of the activation function.
     */
    default String getName() {
        return getClass().getSimpleName().replaceAll("Activation", "");
    }
}
