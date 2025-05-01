package org.brain4j.core.layer.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.structure.StatesCache;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;

import java.io.DataInputStream;
import java.io.DataOutputStream;

/**
 * Represents a normalization layer, used to normalize the inputs and improve training.
 */
public class LayerNorm extends Layer {

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
    public Tensor forward(StatesCache cache, Layer lastLayer, Tensor input, boolean training) {
        return normalize(input);
    }

    public Tensor normalize(Tensor input) {
        Tensor normalized = input.clone();

        int rows = input.shape()[0];

        for (int i = 0; i < rows - 1; i++) {
            Range range = new Range(i, i + 1);

            Tensor token = input.slice(range).vector();
            Tensor normalizedToken = normalize1D(token);

            for (int j = 0; j < normalizedToken.elements(); j++) {
                normalized.set(normalizedToken.get(j), i, j);
            }
        }

        return normalized;
    }

    public Tensor normalize1D(Tensor input) {
        double mean = input.mean();
        double variance = input.variance();
        double std = Math.sqrt(variance + epsilon);
        return input.minus(mean).div(std);
    }
}
