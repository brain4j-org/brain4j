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
        super.serialize(stream);
        stream.writeInt(vocabularySize);
        stream.writeDouble(temperature);

        outProjectionWeights.serialize(stream);
    }

    @Override
    public void deserialize(DataInputStream stream) throws Exception {
        super.deserialize(stream);
        this.vocabularySize = stream.readInt();
        this.temperature = stream.readDouble();
        this.outProjectionWeights = Tensors.zeros(0).deserialize(stream);
    }

    @Override
    public Tensor computeLoss(StatesCache cache, Tensor targets, Tensor outputs, LossFunction lossFunction) {
        Tensor input = cache.getInputTensor(this);

        int rows = input.shape()[0];
        Range range = new Range(rows - 1, rows);

        Tensor delta = outputs.minus(targets.vector()).reshape(1, vocabularySize);
        Tensor last = input.slice(range).reshape(1, dimension); // last token [1, dimension]

        Tensor transposedWeights = outProjectionWeights.transpose(); // [vocab_size, dimension]
        Tensor gradW = optimizer.optimize(this, delta, last);

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

        int rows = input.shape()[0];

        Tensor last = input.slice(new Range(rows - 1, rows));
        Tensor logits = last.matmul(outProjectionWeights).softmax(temperature);

        cache.setOutputTensor(this, logits);

        return logits.vector();
    }

    @Override
    public int getTotalParams() {
        return outProjectionWeights.elements();
    }
}
