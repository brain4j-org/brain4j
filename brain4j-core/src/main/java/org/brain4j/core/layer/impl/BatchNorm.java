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
    public String getLayerName() {
        return "Batch Norm";
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
    public Tensor forward(StatesCache cache, Tensor input, boolean training) {
        int batchSize = input.shape()[0];

        Tensor transposed = input.transpose(); // [dimension, batch_size]
        Tensor result = transposed.clone();

        for (int i = 0; i < batchSize; i++) {
            Range range = new Range(i, i + 1);

            Tensor token = transposed.slice(range).vector(); // [batch_size]
            Tensor normalized = normalize1D(token);

            for (int j = 0; j < normalized.elements(); j++) {
                result.set(normalized.get(j), i, j);
            }
        }

        return result.transpose();
    }

    public Tensor normalize1D(Tensor input) {
        double mean = input.mean();
        double variance = input.variance();
        double std = Math.sqrt(variance + epsilon);
        return input.minus(mean).div(std);
    }
}
