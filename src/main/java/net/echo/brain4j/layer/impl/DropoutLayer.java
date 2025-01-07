package net.echo.brain4j.layer.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.threading.NeuronCacheHolder;

import java.util.List;

/**
 * Represents a Dropout layer for regularization in a neural network.
 * <p>
 * This layer randomly sets a fraction of the input neurons to zero
 * during training to mitigate overfitting.
 */
public class DropoutLayer extends Layer {

    private final double dropout;

    /**
     * Constructs a Dropout layer.
     *
     * @param dropout the dropout rate (0 < dropout <= 1), specifying the
     *                proportion of neurons to deactivate during training.
     * @throws IllegalArgumentException if dropout is not in the range (0, 1].
     */
    public DropoutLayer(double dropout) {
        super(0, Activations.LINEAR);

        if (dropout <= 0 || dropout > 1) {
            throw new IllegalArgumentException("Dropout must be between 0 and 1");
        }

        this.dropout = dropout;
    }

    public void process(NeuronCacheHolder cacheHolder, List<Neuron> neurons) {
        for (Neuron neuron : neurons) {
            if (Math.random() < dropout) {
                neuron.setValue(cacheHolder, 0);
            }
        }
    }

    public void backward(NeuronCacheHolder cacheHolder, List<Neuron> neurons) {
        for (Neuron neuron : neurons) {
            neuron.setValue(cacheHolder, neuron.getValue(cacheHolder) * (1.0 - dropout));
        }
    }

    /**
     * Gets the dropout rate.
     *
     * @return the dropout rate.
     */
    public double getDropout() {
        return dropout;
    }
}