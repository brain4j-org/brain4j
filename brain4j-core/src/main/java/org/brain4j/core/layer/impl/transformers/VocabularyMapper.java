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
    public String getLayerName() {
        return "Vocabulary Mapper";
    }

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        super.serialize(stream);
        stream.writeInt(vocabularySize);
        stream.writeInt(dimension);
        stream.writeDouble(temperature);

        outProjectionWeights.serialize(stream);
    }

    @Override
    public void deserialize(DataInputStream stream) throws Exception {
        super.deserialize(stream);
        this.vocabularySize = stream.readInt();
        this.dimension = stream.readInt();
        this.temperature = stream.readDouble();
        this.outProjectionWeights = Tensors.zeros(0).deserialize(stream);
    }

    @Override
    public Tensor computeLoss(StatesCache cache, Tensor targets, Tensor outputs, LossFunction lossFunction) {
        Tensor input = cache.getInputTensor(this); // [batch_size, seq_len, dimension]

        int[] shape = input.shape();

        int batchSize = shape[0];
        int seqLength = shape[1];

        Tensor delta = outputs.minus(targets); // [batch_size, vocab_size]
        Range range = new Range(seqLength - 1, seqLength);

        for (int i = 0; i < batchSize; i++) {
            Tensor batch = input.slice(new Range(i, i + 1)) // [1, seq_len, dimension]
                    .reshape(seqLength, dimension); // [seq_len, dimension]
            Tensor last = batch.slice(range); // [1, dimension]

            Tensor gradW = optimizer.optimize(this, delta, last);
            outProjectionWeights.sub(gradW);
        }

        /* TODO: Reimplement this with batches
        Tensor transposedWeights = outProjectionWeights.transpose(); // [vocab_size, dimension]

        Tensor gradient = delta.matmul(transposedWeights); // [1, dimension]
        Tensor deltaFull = Tensors.zeros(rows, dimension); // [sequence_length, dimension]

        for (int i = 0; i < gradient.elements(); i++) {
            deltaFull.set(gradient.get(0, i), rows - 1, i);
        }
        return deltaFull;
        */

        return null;
    }

    @Override
    public Tensor forward(StatesCache cache, Tensor input, boolean training) {
        int[] shape = input.shape();

        if (shape.length < 2) {
            throw new IllegalArgumentException(
                    "Input must have the shape [batch_size, seq_len, dimension] or [seq_len, dimension]!"
            );
        }

        if (shape.length == 2) {
            input = input.reshape(1, shape[0], shape[1]);
        }

        int batchSize = shape[0];
        int seqLength = shape[1];
        int dimension = shape[2];

        cache.setInputTensor(this, input);

        Tensor result = Tensors.create(batchSize, vocabularySize);

        for (int i = 0; i < batchSize; i++) {
            Tensor batch = input.slice(new Range(i, i + 1))
                    .reshape(seqLength, dimension);

            Tensor last = batch.slice(new Range(seqLength - 1, seqLength)); // [1, dimension]
            Tensor logits = last.matmul(outProjectionWeights)
                    .softmax(temperature)
                    .vector(); // [vocab_size]

            for (int j = 0; j < logits.elements(); j++) {
                result.set(logits.get(j), i, j);
            }
        }

        cache.setOutputTensor(this, result);
        return result;
    }

    @Override
    public int getTotalWeights() {
        return outProjectionWeights.elements();
    }
}
