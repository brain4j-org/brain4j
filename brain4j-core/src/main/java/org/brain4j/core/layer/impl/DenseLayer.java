package org.brain4j.core.layer.impl;

import jdk.jshell.spi.SPIResolutionException;
import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.core.clipper.impl.HardClipper;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.util.Random;

/**
 * Implementation of a dense (or fully connected) layer.
 * @author xEcho1337
 */
public class DenseLayer extends Layer {

    private final int dimension;

    /**
     * Constructs a new instance of a dense layer.
     * @param dimension the dimension of the output
     * @param activation the activation function to use
     */
    public DenseLayer(int dimension, Activations activation) {
        this(dimension, activation, new HardClipper(5));
    }

    /**
     * Constructs a new instance of a dense layer.
     * @param dimension the dimension of the output
     * @param activation the activation function to use
     * @param clipper the gradient clip function to use
     */
    public DenseLayer(int dimension, Activations activation, GradientClipper clipper) {
        super(activation.getFunction(), clipper);
        this.dimension = dimension;
    }

    @Override
    public void connect(Layer previous) {
        if (previous == null) return;
        // Shape: [output_size, input_size]
        this.weights = Tensors.create(dimension, previous.size()).withGrad();
        this.bias = Tensors.create(dimension).withGrad();
    }

    @Override
    public void initWeights(Random generator, double bound) {
        this.weights.map(x -> (2 * generator.nextDouble() - 1) * bound);
        this.bias.map(x -> (2 * generator.nextDouble() - 1) * bound);
    }

    @Override
    public Tensor forward(StatesCache cache, Tensor input, int index, boolean training) {
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
        Tensor output = input
            .matmulGrad(weights.transpose())
            .addGrad(bias)
            .activateGrad(activation);

        cache.setInput(index, input);
        cache.setOutput(index, output);

        return output;
    }

    @Override
    public int size() {
        return dimension;
    }
}
