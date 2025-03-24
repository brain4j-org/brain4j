package net.echo.brain4j.transformers.vocabulary;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

import java.util.List;
import java.util.Random;

public class VocabularyMapper extends Layer<Tensor, Tensor> {

    private final Tensor outProjectionWeights;
    private final int vocabularySize;
    private final double temperature;

    public VocabularyMapper(int vocabularySize, int dimension, double temperature) {
        this.vocabularySize = vocabularySize;
        this.outProjectionWeights = TensorFactory.random(dimension, vocabularySize); // TODO: matmul support for 1d tensors
        this.temperature = Math.max(1e-15, temperature);
    }

    @Override
    public void init(Random generator) {
    }

    @Override
    public Tensor computeLoss(StatesCache cache, Tensor targets, Tensor outputs, LossFunction lossFunction) {
        Tensor delta = outputs.clone().sub(targets);

        // delta as a 1 x vocab_size matrix
        Tensor gradZ = delta.reshape(1, vocabularySize);
        Tensor gradW = cache.getOutputTensor(this)
                .transpose()
                .matmul(gradZ)
                .mul(optimizer.getLearningRate());

        outProjectionWeights.sub(gradW);
        return delta;
    }

    @Override
    public Tensor forward(StatesCache cache, Layer<?, ?> lastLayer, Tensor input) {
        int columns = input.shape()[1];

        cache.setInputTensor(this, input);

        List<Tensor> tokens = TensorFactory.toList(input);

        Tensor last = tokens.getLast();
        Tensor reshaped = last.reshape(1, columns);

        cache.setOutputTensor(this, reshaped);

        return reshaped
                .matmul(outProjectionWeights)
                .reshape(vocabularySize)
                .softmax(temperature);
    }

    @Override
    public int getTotalParams() {
        return outProjectionWeights.elements();
    }
}
