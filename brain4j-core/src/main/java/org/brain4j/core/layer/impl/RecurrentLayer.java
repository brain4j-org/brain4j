package org.brain4j.core.layer.impl;

import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.util.Random;

/**
 * Implementation of a recurrent layer.
 * @apiNote This implementation is not completed and doesn't support training yet.
 * @author xEcho1337
 * @since 3.0
 */
public class RecurrentLayer extends Layer {

    private final int dimension;
    private final int hiddenDimension;
    private Tensor inputWeights;
    private Tensor hiddenWeights;
    private Tensor hiddenStateBias;

    /**
     * Constructs a new recurrent layer instance.
     *
     * @param dimension the dimension of the output
     * @param hiddenDimension the dimension of the hidden states
     * @param activation the activation function
     */
    public RecurrentLayer(int dimension, int hiddenDimension, Activations activation) {
        this.dimension = dimension;
        this.hiddenDimension = hiddenDimension;
        this.activation = activation.getFunction();
        this.weightInit = this.activation.defaultWeightInit();
    }

    @Override
    public Layer connect(Layer previous) {
        if (previous == null) return this;

        this.inputWeights = Tensors.zeros(hiddenDimension, previous.size()).withGrad();
        this.hiddenWeights = Tensors.zeros(hiddenDimension, hiddenDimension).withGrad();
        this.weights = Tensors.zeros(dimension, hiddenDimension).withGrad();
        this.bias = Tensors.zeros(dimension).withGrad();
        this.hiddenStateBias = Tensors.zeros(hiddenDimension).withGrad();

        return this;
    }

    @Override
    public void initWeights(Random generator, int input, int output) {
        this.inputWeights.map(x -> weightInit.generate(generator, input, output));
        this.hiddenWeights.map(x -> weightInit.generate(generator, input, output));
        this.weights.map(x -> weightInit.generate(generator, input, output));
        this.bias.map(x -> weightInit.generate(generator, input, output));
        this.hiddenStateBias.map(x -> weightInit.generate(generator, input, output));
    }

    @Override
    public Tensor forward(ForwardContext context) {
        Tensor input = context.input();
        StatesCache cache = context.cache();
        int index = context.index();

        int batchSize = input.shape()[0];
        int inputDim = input.shape()[1];
        int expectedDim = inputWeights.shape()[1];

        if (inputDim != expectedDim) {
            throw new IllegalArgumentException(
                "Input dimension mismatch: " + inputDim + " != " + expectedDim
            );
        }

        Tensor previousState = cache.hiddenState(index);

        if (previousState == null) {
            previousState = Tensors.zeros(batchSize, hiddenDimension);
        }

        // [batch_size, hidden_size]
        Tensor projectedInput = input.matmulGrad(inputWeights.transpose());
        // [1, hidden_size]
        Tensor projectedHiddenState = previousState.matmulGrad(hiddenWeights);
        Tensor hiddenState = projectedInput
            .addGrad(projectedHiddenState) // [batch_size, hidden_size]
            .addGrad(hiddenStateBias) // [batch_size, hidden_size]
            .activateGrad(activation);

        // [batch_size, output_size]
        Tensor output = hiddenState.matmulGrad(weights.transpose()).addGrad(bias);

        cache.setInput(index, input);
        cache.setOutput(index, output);
        cache.setHiddenState(index, hiddenState);

        return output;
    }

    @Override
    public int size() {
        return dimension;
    }
}
