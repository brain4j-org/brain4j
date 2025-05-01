package org.brain4j.core.layer.impl.transformers;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.structure.StatesCache;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;
import org.brain4j.math.tensor.index.Range;

import java.io.DataInputStream;
import java.io.DataOutputStream;

public class VocabularyMapper extends Layer {

    private Tensor outProjectionWeights;
    private int vocabularySize;
    private int dimension;
    private double temperature;

    private VocabularyMapper() {
    }

    public VocabularyMapper(int vocabularySize, int dimension, double temperature) {
        this.vocabularySize = vocabularySize;
        this.dimension = dimension;
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
        Tensor input = cache.getInputTensor(this);
        Tensor output = cache.getOutputTensor(this);

        Tensor delta = outputs.minus(targets.vector()).reshape(1, vocabularySize);
        int rows = output.shape()[0];

        Range range = new Range(rows - 1, rows);
        Tensor last = input.slice(range).reshape(1, dimension); // last token [1, dimension]

        Tensor transposedWeights = outProjectionWeights.transpose(); // [vocab_size, dimension]
        Tensor gradW = last.transpose() // [dimension, 1]
                .matmul(delta) // [dimension, vocab_size]
                .mul(optimizer.getLearningRate());

        outProjectionWeights.sub(gradW);

        Tensor gradient = delta.matmul(transposedWeights); // [1, dimension]
        Tensor deltaFull = Tensors.zeros(rows, dimension); // [sequence_length, dimension]

        for (int i = 0; i < gradient.elements(); i++) {
            deltaFull.set(gradient.get(0, i), rows - 1, i);
        }

        return deltaFull;
    }

    @Override
    public Tensor forward(StatesCache cache, Layer lastLayer, Tensor input, boolean training) {
        cache.setInputTensor(this, input);

        Tensor logits = input.matmul(outProjectionWeights)
                .softmax(temperature);

        int rows = logits.shape()[0];

        Range range = new Range(rows - 1, rows);
        Tensor last = logits.slice(range);

        cache.setOutputTensor(this, logits);

        return last.vector();
    }

    @Override
    public int getTotalParams() {
        return outProjectionWeights.elements();
    }
}
