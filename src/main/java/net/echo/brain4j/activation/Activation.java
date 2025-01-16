package net.echo.brain4j.activation;

import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.threading.NeuronCacheHolder;
import net.echo.brain4j.utils.Vector;

import java.util.List;

public interface Activation {

    /**
     * Activate a single scalar value (e.g. ReLU, Sigmoid, Tanh, etc.).
     * If the activation is inherently vector-based (like Softmax),
     * caller may throw an UnsupportedOperationException here.
     */
    double activate(double input);

    /**
     * Activate a vector of values.
     * Return a new array containing the activated values.
     */
    double[] activate(double[] input);

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
     * For purely vector-based activations (Softmax), caller might
     * throw an UnsupportedOperationException here.
     */
    double getDerivative(double input);

    /**
     * Get the derivative (vector) of the activation at a vector of values.
     * For purely vector-based activations like Softmax.
     *
     * @throws UnsupportedOperationException if the activation does not support this operation
     */
    default Vector getDerivative(Vector input) {
        Vector result = new Vector(input.size());

        for (int i = 0; i < input.size(); i++) {
            result.set(i, getDerivative(input.get(i)));
        }

        return result;
    }

    /**
     * Get the full Jacobian matrix (n x n) of the activation for a
     * vector of outputs. Some activations require
     * a matrix derivative. For scalar-based activations, this can
     * throw an UnsupportedOperationException or return a 1x1 matrix.
     *
     * @param outputs The output vector of the activation function
     *                (typically already computed from 'activate').
     * @return An n x n matrix representing d(output)/d(input).
     */
    double[][] getDerivativeMatrix(double[] outputs);

    /**
     * Caller applies this activation function to the values of a list of Neurons.
     */
    void apply(NeuronCacheHolder cacheHolder, List<Neuron> neurons);
}
