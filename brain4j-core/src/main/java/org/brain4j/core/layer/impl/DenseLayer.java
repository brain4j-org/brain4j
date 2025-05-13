package org.brain4j.core.layer.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.util.Random;

public class DenseLayer extends Layer {

    private final int units;

    public DenseLayer(int units, Activation activation) {
        super(activation);
        this.units = units;
    }

    @Override
    public void connect(Layer previous) {
        if (previous == null) return;
        // Shape: [output_size, input_size]
        this.weights = Tensors.create(units, previous.size());
        this.bias = Tensors.create(units);
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
        Tensor output = input.matmulWithGrad(weights.transpose());

        // TODO: broadcast adding the bias

        Tensor activated = output.activateWithGrad(activation);

        cache.setInput(index, input);
        cache.setOutput(index, activated);

        return activated;
    }

    @Override
    public int size() {
        return units;
    }
}
