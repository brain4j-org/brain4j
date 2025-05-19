package org.brain4j.core.layer.impl;

import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.core.clipper.impl.HardClipper;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.util.Random;

/**
 * Implementation of a recurrent layer.
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
     * @param dimension the dimension of the output
     * @param hiddenDimension the dimension of the hidden states
     * @param activation the activation function
     */
    public RecurrentLayer(int dimension, int hiddenDimension, Activations activation) {
        this(dimension, hiddenDimension, activation.getFunction(), new HardClipper(5));
    }

    /**
     * Constructs a new recurrent layer instance.
     * @param dimension the dimension of the output
     * @param hiddenDimension the dimension of the hidden states
     * @param activation the activation function
     * @param clipper the gradient clip function
     */
    public RecurrentLayer(int dimension, int hiddenDimension, Activation activation, GradientClipper clipper) {
        super(activation, clipper);
        this.dimension = dimension;
        this.hiddenDimension = hiddenDimension;
    }

    @Override
    public void connect(Layer previous) {
        if (previous == null) return;

        this.inputWeights = Tensors.create(hiddenDimension, previous.size()).withGrad();
        this.hiddenWeights = Tensors.create(hiddenDimension, hiddenDimension).withGrad();
        this.weights = Tensors.create(dimension, hiddenDimension).withGrad();
        this.bias = Tensors.create(dimension).withGrad();
        this.hiddenStateBias = Tensors.create(hiddenDimension).withGrad();
    }

    @Override
    public void initWeights(Random generator, double bound) {
        this.inputWeights.map(x -> (2 * generator.nextDouble() - 1) * bound);
        this.hiddenWeights.map(x -> (2 * generator.nextDouble() - 1) * bound);
        this.weights.map(x -> (2 * generator.nextDouble() - 1) * bound);
        this.bias.map(x -> (2 * generator.nextDouble() - 1) * bound);
        this.hiddenStateBias.map(x -> (2 * generator.nextDouble() - 1) * bound);
    }

    @Override
    public Tensor forward(StatesCache cache, Tensor input, int index, boolean training) {
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
            previousState = Tensors.create(batchSize, hiddenDimension);
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
