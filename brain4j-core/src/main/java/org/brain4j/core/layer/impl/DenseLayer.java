package org.brain4j.core.layer.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.Tensors;
import org.brain4j.core.activation.Activations;
import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;

import java.util.Arrays;
import java.util.Random;

/**
 * Implementation of a fully connected (dense) neural network layer.
 * <p>
 * This layer performs a linear transformation on the input tensor,
 * followed by the application of a specified activation function.
 * </p>
 * <p>Inputs are expected to have the shape <code>[batch_size, input_size]</code>,
 * outputs have the shape <code>[batch_size, dimension]</code> where <code>dimension</code>
 * is the amount of neurons in this layer.
 * </p>
 * Weights are represented with the following shapes:
 * <ul>
 *   <li><code>weights</code> has shape <code>[input_size, output_size]</code></li>
 *   <li><code>bias</code> has shape <code>[output_size]</code></li>
 * </ul>
 * @author xEcho1337
 * @since 3.0
 */
public class DenseLayer extends Layer {

    private final int dimension;

    /**
     * Constructs a new instance of a dense layer with a linear activation.
     * @param dimension the dimension of the output
     */
    public DenseLayer(int dimension) {
        this(dimension, Activations.LINEAR);
    }

    /**
     * Constructs a new instance of a dense layer.
     * @param dimension the dimension of the output
     * @param activation the activation function
     */
    public DenseLayer(int dimension, Activations activation) {
        this.dimension = dimension;
        this.activation = activation.function();
    }

    @Override
    public Layer connect(Layer previous) {
        if (previous == null) return this;

        // Shape: [input_size, output_size]
        this.weights = Tensors.zeros(previous.size(), dimension).withGrad();
        this.bias = Tensors.zeros(dimension).withGrad();

        return this;
    }

    @Override
    public void initWeights(Random generator, int input, int output) {
        if (input == 0) return;

        this.weights.map(x -> weightInit.generate(generator, input, output));
        this.bias.map(x -> weightInit.generate(generator, input, output));
    }

    @Override
    public Tensor forward(ForwardContext context) {
        Tensor input = context.input();
        StatesCache cache = context.cache();

        if (weights == null) return input;

        if (!validateInput(input)) {
            throw new IllegalArgumentException(
                "Input dimension mismatch. Got: " + Arrays.toString(input.shape()) + " Expected: " + weights.shape()[0]
            );
        }

        // Shape: [batch_size, output_size]
        Tensor output = input
            .matmulGrad(weights)
            .addGrad(bias);

        cache.setPreActivation(this, output);
        return output.activateGrad(activation);
    }

    @Override
    public int size() {
        return dimension;
    }

    @Override
    public boolean validateInput(Tensor input) {
        int[] shape = input.shape();
        int[] weightsShape = weights.shape();
        return shape[shape.length - 1] == weightsShape[0];
    }
}
