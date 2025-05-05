package org.brain4j.core.layer.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.structure.StatesCache;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;

import java.io.DataInputStream;
import java.io.DataOutputStream;

public class BatchNorm extends Layer {

    private double epsilon;

    /**
     * Constructs a layer normalization instance with a default epsilon.
     */
    public BatchNorm() {
        this(1e-5);
    }

    /**
     * Constructs a layer normalization instance with an epsilon.
     * @param epsilon the epsilon used to avoid division by zero
     */
    public BatchNorm(double epsilon) {
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
    public Tensor forward(StatesCache cache, Layer lastLayer, Tensor input, boolean training) {
        int batchSize = input.shape()[0];
        Tensor transposed = input.transpose(); // [dimension, batch_size]

        for (int i = 0; i < batchSize; i++) {
            Tensor feature = transposed.slice(new Range(i, i + 1)); // [1, batch_size]

            double mean = feature.mean();
            double variance = feature.variance();
            double std = Math.sqrt(variance + epsilon);
            // TODO: Finish this
        }

        return input;
    }

    @Override
    public Tensor backward(StatesCache cache, Layer previous, Tensor delta) {
        return delta;
    }
}
