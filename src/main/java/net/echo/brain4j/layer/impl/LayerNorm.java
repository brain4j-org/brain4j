package net.echo.brain4j.layer.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.threading.NeuronCacheHolder;
import net.echo.brain4j.utils.Vector;

import java.util.List;

public class LayerNorm extends Layer {

    private final double epsilon;

    public LayerNorm() {
        this(1e-5);
    }

    public LayerNorm(double epsilon) {
        super(0, Activations.LINEAR);
        this.epsilon = epsilon;
    }

    @Override
    public boolean canPropagate() {
        return false;
    }

    @Override
    public void applyFunction(NeuronCacheHolder cacheHolder, Layer previous) {
        List<Neuron> inputs = previous.getNeurons();

        double mean = calculateMean(cacheHolder, inputs);
        double variance = calculateVariance(cacheHolder, inputs, mean);

        for (Neuron input : inputs) {
            double value = input.getValue(cacheHolder);
            double normalized = (value - mean) / Math.sqrt(variance + epsilon);

            input.setValue(cacheHolder, normalized);
        }
    }

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

    private double calculateMean(NeuronCacheHolder cacheHolder, List<Neuron> inputs) {
        double sum = 0.0;

        for (Neuron value : inputs) {
            sum += value.getValue(cacheHolder);
        }

        return sum / inputs.size();
    }

    private double calculateVariance(NeuronCacheHolder cacheHolder, List<Neuron> inputs, double mean) {
        double sum = 0.0;

        for (Neuron value : inputs) {
            sum += Math.pow(value.getValue(cacheHolder) - mean, 2);
        }

        return sum / inputs.size();
    }
}
