package org.brain4j.core.layer.impl;

import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.core.clipper.impl.HardClipper;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.util.Random;

public class DenseLayer extends Layer {

    private final int neurons;

    public DenseLayer(int neurons, Activations activation) {
        this(neurons, activation, new HardClipper(5));
    }

    public DenseLayer(int neurons, Activations activation, GradientClipper clipper) {
        super(activation.getFunction(), clipper);
        this.neurons = neurons;
    }

    @Override
    public void connect(Layer previous) {
        if (previous == null) return;
        // Shape: [output_size, input_size]
        this.weights = Tensors.create(neurons, previous.size()).withGrad();
        this.bias = Tensors.create(neurons).withGrad();
    }

    @Override
    public void initWeights(Random generator, double bound) {
        this.weights.map(x -> (2 * generator.nextDouble() - 1) * bound);
        this.bias.map(x -> (2 * generator.nextDouble() - 1) * bound);
    }

    // Input has the shape [batch_size, input_size]
    @Override
    public Tensor forward(StatesCache cache, Tensor input, int index, boolean training) {
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
        return neurons;
    }
}
