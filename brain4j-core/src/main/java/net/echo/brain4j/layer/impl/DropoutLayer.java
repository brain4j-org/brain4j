package net.echo.brain4j.layer.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.math4j.math.tensor.Tensor;

import java.io.DataInputStream;
import java.io.DataOutputStream;

/**
 * Represents a Dropout layer, used to mitigate overfitting
 * by randomly deactivating a fraction of the input neurons.
 */
public class DropoutLayer extends Layer<Tensor, Tensor> {

    private double dropout;

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
    public void serialize(DataOutputStream stream) throws Exception {
        super.serialize(stream);
        stream.writeDouble(dropout);
    }

    @Override
    public void deserialize(DataInputStream stream) throws Exception {
        super.deserialize(stream);
        this.dropout = stream.readDouble();
    }

    @Override
    public boolean canPropagate() {
        return false;
    }

    @Override
    public Tensor forward(StatesCache cache, Layer<?, ?> lastLayer, Tensor input) {
        if (!(lastLayer instanceof DenseLayer)) {
            throw new UnsupportedOperationException("Layer before must be a dense layer!");
        }

        if (input.dimension() != 1) {
            throw new UnsupportedOperationException("Only 1D tensors are supported!");
        }

        for (int i = 0; i < input.elements(); i++) {
            if (Math.random() > dropout) continue;

            input.set(0, i);
        }

        return input;
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