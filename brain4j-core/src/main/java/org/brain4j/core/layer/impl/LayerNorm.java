package org.brain4j.core.layer.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.structure.StatesCache;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;
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
    public LayerNorm(int input) {
        this(input, 1e-5);
    }

    /**
     * Constructs a layer normalization instance with an epsilon.
     * @param epsilon the epsilon used to avoid division by zero
     */
    public LayerNorm(int input, double epsilon) {
        super(input, Activations.LINEAR);
        this.epsilon = epsilon;
        this.weights = Tensors.ones(input);
    }

    @Override
    public String getLayerName() {
        return "Layer Norm";
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
        int batchSize = input.shape()[0];
        Tensor result = input.clone();

        for (int i = 0; i < batchSize; i++) {
            Range range = new Range(i, i + 1);

            Tensor token = input.slice(range).vector();
            Tensor row = normalize1D(token).mul(weights).add(bias);

            for (int j = 0; j < row.elements(); j++) {
                result.set(row.get(j), i, j);
            }
        }

        return result;
    }

    public Tensor normalize1D(Tensor input) {
        double mean = input.mean();
        double variance = input.variance();
        double std = Math.sqrt(variance + epsilon);
        return input.minus(mean).div(std);
    }
}
