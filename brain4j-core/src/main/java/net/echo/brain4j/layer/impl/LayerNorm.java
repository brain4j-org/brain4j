package net.echo.brain4j.layer.impl;

import com.google.gson.internal.bind.util.ISO8601Utils;
import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.index.Range;
import net.echo.math4j.math.vector.Vector;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.Arrays;
import java.util.List;

/**
 * Represents a normalization layer, used to normalize the inputs and improve training.
 */
public class LayerNorm extends Layer<Vector, Vector> {

    private double epsilon;

    /**
     * Constructs a layer normalization instance with a default epsilon.
     */
    public LayerNorm() {
        this(1e-5);
    }

    /**
     * Constructs a layer normalization instance with an epsilon.
     * @param epsilon the epsilon used to avoid division by zero
     */
    public LayerNorm(double epsilon) {
        super(0, Activations.LINEAR);
        this.epsilon = epsilon;
    }

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        super.serialize(stream);
        stream.writeDouble(epsilon);
    }

    @Override
    public void deserialize(DataInputStream stream) throws Exception {
        super.deserialize(stream);
        this.epsilon = stream.readDouble();
    }

    @Override
    public boolean canPropagate() {
        return false;
    }

    @Override
    public void applyFunction(StatesCache cacheHolder, Layer<?, ?> previous) {
        List<Neuron> inputs = previous.getNeurons();

        double mean = calculateMean(cacheHolder, inputs);
        double variance = calculateVariance(cacheHolder, inputs, mean);

        for (Neuron input : inputs) {
            double value = input.getValue(cacheHolder);
            double normalized = (value - mean) / Math.sqrt(variance + epsilon);

            input.setValue(cacheHolder, normalized);
        }
    }

    /**
     * Normalizes a vector with a mean of zero and variance of one.
     *
     * @param input the input vector to normalize
     * @return the normalized output vector
     */
    public Vector normalize(Vector input) {
        double mean = input.mean();
        double variance = input.variance(mean);

        double denominator = Math.sqrt(variance + epsilon);

        for (int i = 0; i < input.size(); i++) {
            double value = input.get(i);
            double normalized = (value - mean) / denominator;

            input.set(i, normalized);
        }

        return input;
    }
    
    /**
     * Normalizes a tensor with a mean of zero and variance of one.
     *
     * @param input the input tensor to normalize
     * @return the normalized output tensor
     */
    public Tensor normalize(Tensor input) {
        Tensor normalized = input.clone();

        int rows = input.shape()[0];
        int columns = input.shape()[1];

        for (int i = 0; i < rows; i++) {
            Range range = new Range(i * columns, (i + 1) * columns);
            Tensor token = input.reshape(columns * rows).slice(range);

            double mean = token.mean();
            double variance = 0.0;

            for (Double value : token) {
                variance += Math.pow(value - mean, 2);
            }

            variance /= token.elements();

            double denominator = Math.sqrt(variance + epsilon);

            for (int j = 0; j < token.elements(); j++) {
                double value = token.get(j);
                double normalizedValue = (value - mean) / denominator;

                normalized.set(normalizedValue, i, j);
            }
        }

        return normalized;
    }
//
//    public Tensor normalize(Tensor input) {
//        double mean = input.mean();
//        double variance = 0.0;
//
//        for (Double value : input) {
//            variance += Math.pow(value - mean, 2);
//        }
//        variance /= input.elements();
//
//        double denominator = Math.sqrt(variance + epsilon);
//
//        Tensor normalized = input.clone();
//
//        return normalized.map(value -> (value - mean) / denominator);
//    }

    private double calculateMean(StatesCache cacheHolder, List<Neuron> inputs) {
        double sum = 0.0;

        for (Neuron value : inputs) {
            sum += value.getValue(cacheHolder);
        }

        return sum / inputs.size();
    }

    private double calculateVariance(StatesCache cacheHolder, List<Neuron> inputs, double mean) {
        double sum = 0.0;

        for (Neuron value : inputs) {
            sum += Math.pow(value.getValue(cacheHolder) - mean, 2);
        }

        return sum / inputs.size();
    }
}
