package net.echo.brain4j.layer.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.utils.Vector;

import java.util.Objects;

/**
 * Represents a Dropout layer, used to mitigate overfitting
 * by randomly deactivating a fraction of the input neurons.
 */
public class DropoutLayer extends Layer<Vector, Vector> {

    private final double dropout;

    /**
     * Constructs a dropout layer instance.
     *
     * @param dropout the dropout rate (0 < dropout < 1), specifying the
     *                proportion of neurons to deactivate during training
     * @throws IllegalArgumentException if dropout is outside the range 0-1
     */
    public DropoutLayer(double dropout) {
        super(0, Activations.LINEAR);

        if (dropout <= 0 || dropout >= 1) {
            throw new IllegalArgumentException("Dropout must be greater than 0 and less than 1!");
        }

        this.dropout = dropout;
    }

    @Override
    public void propagate(StatesCache cacheHolder, Layer<?, ?> previous, Updater updater, Optimizer optimizer) {
        Objects.requireNonNull(previous, "Previous layer is null");

        for (Neuron neuron : previous.getNeurons()) {
            neuron.setValue(cacheHolder, neuron.getValue(cacheHolder) * (1.0 - dropout));
        }
    }

    /**
     * Gets the dropout rate.
     *
     * @return the dropout rate
     */
    public double getDropout() {
        return dropout;
    }
}