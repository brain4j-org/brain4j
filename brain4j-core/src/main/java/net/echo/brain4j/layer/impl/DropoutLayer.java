package net.echo.brain4j.layer.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.StatesCache;
import net.echo.math.tensor.Tensor;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.Random;
import java.util.SplittableRandom;

/**
 * Represents a Dropout layer, it's used to mitigate overfitting by randomly deactivating a fraction of the neurons
 * during training.
 * When inferencing, the output gets scaled by {@code 1 - dropout}.
 */
public class DropoutLayer extends Layer {

    private final Random fastRandom = Random.from(new SplittableRandom());
    private double dropout;

    public DropoutLayer() {
    }

    /**
     * Constructs a dropout layer instance.
     *
     * @param dropout The dropout rate (0 < dropout < 1), specifying the
     *                proportion of neurons to deactivate during training
     * @throws IllegalArgumentException If dropout is outside the range 0-1.
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

    /**
     * Applies the dropout to the input tensor. This method will randomly set to 0 random values from the input tensor
     * during training. Meanwhile it will scale the input tensor by {@code 1 - dropout} during inferencing.
     *
     * @param nextLayer
     * @param input     The input tensor.
     * @param training  If it's called during training.
     * @return The resulting tensor.
     */
    @Override
    public Tensor forward(StatesCache cache, Layer lastLayer, Layer nextLayer, Tensor input, boolean training) {
        if (training) {
            return scale(input);
        }

        if (input.dimension() != 1) {
            throw new UnsupportedOperationException("Only 1D tensors are supported!");
        }

        for (int i = 0; i < input.elements(); i++) {
            if (fastRandom.nextDouble() > dropout) {
                continue;
            }

            input.set(0, i);
        }

        return input;
    }

    /**
     * Scales the input tensor by {@code 1 - input}.
     * @param input The input tensor.
     * @return The scaled tensor.
     */
    public Tensor scale(Tensor input) {
        return input.mul(1 - dropout);
    }

    /**
     * Gets the dropout rate.
     *
     * @return The dropout rate.
     */
    public double getDropout() {
        return dropout;
    }
}