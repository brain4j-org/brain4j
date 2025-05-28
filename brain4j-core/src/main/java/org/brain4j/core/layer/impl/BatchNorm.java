package org.brain4j.core.layer.impl;

import org.brain4j.core.clipper.impl.NoClipper;
import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;
import org.brain4j.math.activation.impl.LinearActivation;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;

/**
 * Implementation of a batch normalization layer, it's used to normalize inputs and improve training.
 * @author xEcho1337
 */
public class BatchNorm extends Layer {

    private final double epsilon;

    /**
     * Constructs a layer normalization instance with a default epsilon of 1e-5.
     */
    public BatchNorm() {
        this(1e-5);
    }

    /**
     * Constructs a new instance of a batch normalization layer.
     * @param epsilon the epsilon used to avoid division by zero
     */
    public BatchNorm(double epsilon) {
        super(new LinearActivation(), new NoClipper());
        this.epsilon = epsilon;
    }

    @Override
    public Tensor forward(ForwardContext context) {
        Tensor input = context.input();
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

    @Override
    public int size() {
        return 0;
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
