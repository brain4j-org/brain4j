package org.brain4j.core.layer.impl;

import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
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
        this.epsilon = epsilon;
    }

    @Override
    public void connect(Layer previous) {
        this.weights = Tensors.ones(previous.size()).withGrad();
        this.bias = Tensors.zeros(previous.size()).withGrad();
    }

    @Override
    public Tensor forward(ForwardContext context) {
        Tensor input = context.input();
        Tensor result = input.clone();

        int batchSize = input.shape()[0];
        int dimension = input.shape()[1];

        float[] inputData = input.data();
        float[] resultData = result.data();

        float[] biasData = bias.data();
        float[] weightsData = weights.data();

        for (int i = 0; i < batchSize; i++) {
            double mean = 0.0;
            double variance = 0.0;

            for (int j = 0; j < dimension; j++) {
                float value = inputData[i * dimension + j];
                mean += value;
            }

            mean /= dimension;

            for (int j = 0; j < dimension; j++) {
                float value = inputData[i * dimension + j];
                double diff = value - mean;
                variance += diff * diff;
            }

            variance /= dimension;

            double std = Math.sqrt(variance + epsilon);

            for (int j = 0; j < dimension; j++) {
                int index = i * dimension + j;
                double value = resultData[index];
                double normalized = (value - mean) / std;

                double scaled = weightsData[j] * normalized + biasData[j];
                resultData[index] = (float) scaled;
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
