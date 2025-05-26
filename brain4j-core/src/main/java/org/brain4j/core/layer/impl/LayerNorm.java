package org.brain4j.core.layer.impl;

import org.brain4j.core.clipper.impl.NoClipper;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;
import org.brain4j.math.activation.impl.LinearActivation;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;
import org.brain4j.math.tensor.index.Range;

/**
 * Implementation of a layer normalization layer, it's used to normalize inputs and improve training.
 * @author xEcho1337
 */
public class LayerNorm extends Layer {

    private final double epsilon;

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
        super(new LinearActivation(), new NoClipper());
        this.epsilon = epsilon;
    }

    @Override
    public void connect(Layer previous, Layer next) {
        this.weights = Tensors.ones(previous.size()).withGrad();
        this.bias = Tensors.zeros(previous.size()).withGrad();
    }

    @Override
    public Tensor forward(StatesCache cache, Tensor input, int index, boolean training) {
        int batchSize = input.shape()[0];

        for (int i = 0; i < batchSize; i++) {
            Range range = new Range(i, i + 1);

            Tensor token = input.slice(range).vector();
            Tensor row = normalize1D(token).mul(weights).add(bias);

            for (int j = 0; j < row.elements(); j++) {
                input.set(row.get(j), i, j);
            }
        }

        return input;
    }

    @Override
    public int size() {
        return weights.elements();
    }

    @Override
    public boolean skipPropagate() {
        return true;
    }

    public Tensor normalize1D(Tensor input) {
        double mean = input.mean();
        double variance = input.variance();
        double std = Math.sqrt(variance + epsilon);
        return input.minus(mean).div(std);
    }
}
