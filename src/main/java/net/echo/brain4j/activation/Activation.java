package net.echo.brain4j.activation;

import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.utils.math.vector.Vector;

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
    default Vector activate(Vector input) {
        Vector output = new Vector(input.size());

        for (int i = 0; i < input.size(); i++) {
            output.set(i, activate(input.get(i)));
        }

        return output;
    }

    /**
     * Get the derivative (scalar) of the activation at a single value.
     */
    double getDerivative(double input);

    /**
     * Get the derivative (vector) of the activation at a vector of values.
     */
    default Vector getDerivative(Vector input) {
        Vector result = new Vector(input.size());

        for (int i = 0; i < input.size(); i++) {
            result.set(i, getDerivative(input.get(i)));
        }

        return result;
    }

    /**
     * Caller applies this activation function to the values of a list of Neurons.
     */
    default void apply(StatesCache cacheHolder, List<Neuron> neurons) {
        for (Neuron neuron : neurons) {
            double output = neuron.getValue(cacheHolder) + neuron.getBias();
            double activated = activate(output);

            neuron.setValue(cacheHolder, activated);
        }
    }
}
