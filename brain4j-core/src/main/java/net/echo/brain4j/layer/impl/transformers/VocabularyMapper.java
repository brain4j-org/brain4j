package net.echo.brain4j.layer.impl.transformers;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.structure.StatesCache;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.Tensors;
import net.echo.math.tensor.index.Range;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.List;

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
        Tensor output = cache.getOutputTensor(this);
        Tensor delta = outputs.minus(targets)
                .reshape(1, vocabularySize); // [1, vocab_size]

        int rows = output.shape()[0];

        Range range = new Range(rows - 1, rows);
        Tensor last = output.slice(range).reshape(1, dimension); // last token [1, dimension]

        Tensor transposedWeights = outProjectionWeights.transpose(); // [vocab_size, dimension]
        Tensor gradW = last.transpose() // [dimension, 1]
                .matmul(delta) // [dimension, vocab_size]
                .mul(optimizer.getLearningRate());

        outProjectionWeights.sub(gradW);

        Tensor gradient = delta.matmul(transposedWeights); // [1, dimension]
        Tensor deltaFull = Tensors.zeros(rows, dimension); // [sequence_length, dimension]

        for (int i = 0; i < gradient.elements(); i++) {
            deltaFull.set(gradient.get(i), rows - 1, i);
        }

        return deltaFull;
    }

    @Override
    public Tensor forward(StatesCache cache, Layer lastLayer, Layer nextLayer, Tensor input, boolean training) {
        cache.setInputTensor(this, input);

        Tensor logits = input.matmul(outProjectionWeights)
                .softmax(temperature);

        int rows = logits.shape()[0];

        Range range = new Range(rows - 1, rows);
        Tensor last = logits.slice(range);

        cache.setOutputTensor(this, logits);

        return last.reshape(vocabularySize);
    }

    @Override
    public int getTotalParams() {
        return outProjectionWeights.elements();
    }
}
