package org.brain4j.core.layer.impl;

import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

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
 *   <li><code>weights</code> has shape <code>[output_size, input_size]</code></li>
 *   <li><code>bias</code> has shape <code>[output_size]</code></li>
 * </ul>
 * @author xEcho1337
 * @since 3.0
 */
public class DenseLayer extends Layer {

    private final int dimension;

    /**
     * Constructs a new instance of a dense layer.
     * @param dimension the dimension of the output
     * @param activation the activation function
     */
    public DenseLayer(int dimension, Activations activation) {
        this.dimension = dimension;
        this.activation = activation.getFunction();
    }

    @Override
    public Layer connect(Layer previous) {
        if (previous == null) return this;

        // Shape: [output_size, input_size]
        this.weights = Tensors.zeros(dimension, previous.size()).withGrad();
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
        int index = context.index();

        if (weights == null) return input;

        // Input shape: [batch_size, input_size]
        int inputDim = input.shape()[1];
        int expectedDim = weights.shape()[1];

        if (inputDim != expectedDim) {
            throw new IllegalArgumentException(
                "Input dimension mismatch: " + inputDim + " != " + expectedDim
            );
        }

        // Shape: [batch_size, output_size]
        Tensor transposed = weights.transpose();
        Tensor output = input
            .matmulGrad(transposed)
            .addGrad(bias);

        cache.setInput(index, input);
        cache.setPreActivation(index, output);

        Tensor activated = output.activateGrad(activation);
        cache.setOutput(index, activated);

        return activated;
    }

    @Override
    public int size() {
        return dimension;
    }

    @Override
    public boolean validateInput(Tensor input) {
        return input.shape()[1] == dimension;
    }
}
