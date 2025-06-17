package org.brain4j.core.layer.impl.transformer;

import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.activation.impl.SoftmaxActivation;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.util.Arrays;
import java.util.Random;

public class OutVocabLayer extends Layer {

    private final int vocabSize;
    private final int dimension;
    private final double temperature;

    public OutVocabLayer(int vocabSize, int dimension, double temperature) {
        this.vocabSize = vocabSize;
        this.dimension = dimension;
        this.temperature = Math.max(1e-15, temperature);
        this.activation = new SoftmaxActivation(temperature);
    }

    @Override
    public Layer connect(Layer previous) {
        this.weights = Tensors.zeros(dimension, vocabSize);
        return this;
    }

    @Override
    public void initWeights(Random generator, int input, int output) {
        this.weights.map(x -> weightInit.generate(generator, input, output));
    }

    @Override
    public Tensor forward(ForwardContext context) {
        Tensor input = context.input();
        int[] shape = input.shape();

        if (shape.length != 3) {
            throw new IllegalArgumentException(
                    "Expected input with shape [batch_size, seq_length, dimension], got: " + Arrays.toString(shape)
            );
        }

        StatesCache cache = context.cache();
        int index = context.index();

        int batchSize = shape[0];
        int lastToken = shape[1] - 1;
        int dimension = shape[2];

        Tensor result = Tensors.zeros(batchSize, dimension).withGrad();

        for (int b = 0; b < batchSize; b++) {
            for (int d = 0; d < dimension; d++) {
                double value = input.get(b, lastToken, d);
                result.set(value, b, d);
            }
        }

        Tensor output = result.matmulGrad(weights);
        Tensor activated = output.activateGrad(activation);

        cache.setPreActivation(this, output);

        return activated;
    }

    @Override
    public int size() {
        return vocabSize;
    }
}
