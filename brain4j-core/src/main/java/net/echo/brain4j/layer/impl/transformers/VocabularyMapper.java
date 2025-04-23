package net.echo.brain4j.layer.impl.transformers;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.structure.StatesCache;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.Tensors;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.List;

public class VocabularyMapper extends Layer {

    private Tensor outProjectionWeights;
    private int vocabularySize;
    private double temperature;

    private VocabularyMapper() {
    }

    public VocabularyMapper(int vocabularySize, int dimension, double temperature) {
        this.vocabularySize = vocabularySize;
        this.outProjectionWeights = Tensors.random(dimension, vocabularySize);
        this.temperature = Math.max(1e-15, temperature);
    }

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        stream.writeInt(vocabularySize);
        stream.writeDouble(temperature);

        outProjectionWeights.serialize(stream);
    }

    @Override
    public void deserialize(DataInputStream stream) throws Exception {
        this.vocabularySize = stream.readInt();
        this.temperature = stream.readDouble();
        this.outProjectionWeights = Tensors.zeros(0).deserialize(stream);
    }

    @Override
    public Tensor computeLoss(StatesCache cache, Tensor targets, Tensor outputs, LossFunction lossFunction) {
        Tensor delta = outputs.clone().sub(targets);

        // delta as a 1 x vocab_size matrix
        Tensor gradZ = delta.reshape(1, vocabularySize);
        // output shape is 1 x vocab_size
        Tensor gradW = cache.getOutputTensor(this)
                .transpose() // transforms to vocab_size x 1
                .matmul(gradZ)
                .mul(optimizer.getLearningRate());

        outProjectionWeights.sub(gradW);

        return gradZ.matmul(outProjectionWeights.transpose());
    }

    @Override
    public Tensor forward(StatesCache cache, Layer lastLayer, Layer nextLayer, Tensor input, boolean training) {
        int columns = input.shape()[1];

        cache.setInputTensor(this, input);

        List<Tensor> tokens = Tensors.toList(input);

        Tensor last = tokens.getLast();
        Tensor reshaped = last.reshape(1, columns);

        cache.setOutputTensor(this, reshaped);

        return reshaped.matmul(outProjectionWeights)
                .reshape(vocabularySize)
                .softmax(temperature);
    }

    @Override
    public int getTotalParams() {
        return outProjectionWeights.elements();
    }
}
