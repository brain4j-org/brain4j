package org.brain4j.core.layer.impl;

import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.core.clipper.impl.HardClipper;
import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;
import org.brain4j.math.weights.WeightInitialization;

import java.util.Random;

/**
 * Implementation of a dense (or fully connected) layer.
 * @author xEcho1337
 * @since 2.0
 */
public class DenseLayer extends Layer {

    private final int dimension;

    /**
     * Constructs a new instance of a dense layer.
     * @param dimension the dimension of the output
     * @param activation the activation function
     */
    public DenseLayer(int dimension, Activations activation) {
        this(dimension, activation, new HardClipper(5));
    }

    /**
     * Constructs a new instance of a dense layer.
     *
     * @param dimension the dimension of the output
     * @param activation the activation function
     * @param clipper the gradient clip function
     */
    public DenseLayer(int dimension, Activations activation, GradientClipper clipper) {
        this(dimension, activation, clipper, activation.getFunction().defaultWeightInit());
    }

    /**
     * Constructs a new instance of a dense layer.
     *
     * @param dimension the dimension of the output
     * @param activation the activation function
     * @param clipper the gradient clip function
     * @param weightInit the weight initialization function
     */
    public DenseLayer(int dimension, Activations activation, GradientClipper clipper, WeightInitialization weightInit) {
        super(activation.getFunction(), clipper, weightInit);
        this.dimension = dimension;
    }

    @Override
    public void connect(Layer previous, Layer next) {
        super.connect(previous, next);

        if (previous == null) return;

        // Shape: [output_size, input_size]
        this.weights = Tensors.zeros(dimension, previous.size()).withGrad();
        this.bias = Tensors.zeros(dimension).withGrad();
    }

    @Override
    public void initWeights(Random generator, int input, int output) {
        this.weights.map(x -> weightInit.randomValue(generator, input, output));
        this.bias.map(x -> weightInit.randomValue(generator, input, output));
    }

    @Override
    public Tensor forward(ForwardContext context) {
        Tensor input = context.input();
        StatesCache cache = context.cache();
        int index = context.index();
        boolean training = context.training();

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
            .addGrad(bias);

        cache.setInput(index, input);

        if (next instanceof LayerNorm layerNorm) {
            output = layerNorm.forward(new ForwardContext(cache, output, index + 1, training));
        }

        Tensor activated = output.activateGrad(activation);

        cache.setPreActivation(index, output);
        cache.setOutput(index, activated);

        return activated;
    }

    @Override
    public int size() {
        return dimension;
    }
}
