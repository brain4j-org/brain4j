package org.brain4j.core.layer.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.util.Random;

public class DenseLayer extends Layer {

    private final int neurons;

    public DenseLayer(int neurons, Activations activation) {
        super(activation.getFunction());
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
        Tensor output = input.matmulGrad(weights.transpose());

//        int batchSize = output.shape()[0];
//        int elements = output.shape()[1];
//
//        for (int i = 0; i < batchSize; i++) {
//            for (int j = 0; j < elements; j++) {
//                float value = output.get(i, j);
//                float biasValue = bias.get(j);
//
//                output.set(value + biasValue, i, j);
//            }
//        }

        Tensor activated = output.activateGrad(activation);

        cache.setInput(index, input);
        cache.setOutput(index, activated);

        return activated;
    }

    @Override
    public int size() {
        return neurons;
    }
}
