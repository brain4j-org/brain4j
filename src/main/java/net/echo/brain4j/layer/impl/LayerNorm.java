package net.echo.brain4j.layer.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.utils.math.vector.Vector;

import java.io.DataInputStream;
import java.io.DataOutputStream;
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
