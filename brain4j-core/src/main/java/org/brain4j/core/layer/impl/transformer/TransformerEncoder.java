package org.brain4j.core.layer.impl.transformer;

import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.DropoutLayer;
import org.brain4j.core.layer.impl.LayerNorm;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.transformer.attention.MultiHeadAttention;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.device.DeviceType;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;
import org.brain4j.math.tensor.index.Range;

import java.util.Arrays;
import java.util.Random;

public class TransformerEncoder extends Layer {

    protected DenseLayer upProjection;
    protected DenseLayer downProjection;
    protected LayerNorm normalizer;
    protected DropoutLayer dropout;
    protected MultiHeadAttention attention;

    protected int numHeads;
    protected int embeddingDim;

    public TransformerEncoder(int numHeads, int embeddingDim, double dropout) {
        this.numHeads = numHeads;
        this.embeddingDim = embeddingDim;
        this.dropout = new DropoutLayer(dropout);

        this.normalizer = new LayerNorm(embeddingDim);
        this.upProjection = new DenseLayer(embeddingDim * 4, Activations.RELU);
        this.downProjection = new DenseLayer(embeddingDim, Activations.LINEAR);

        this.attention = createAttention(numHeads, embeddingDim);
    }

    public MultiHeadAttention createAttention(int heads, int embeddingDim) {
        return new MultiHeadAttention(heads, embeddingDim);
    }

    @Override
    public Layer connect(Layer previous) {
        this.normalizer.connect(this);
        this.upProjection.connect(this);
        this.downProjection.connect(upProjection);

        return this;
    }

    @Override
    public void initWeights(Random generator, int input, int output) {
        this.normalizer.initWeights(generator, embeddingDim, embeddingDim);
        this.upProjection.initWeights(generator, embeddingDim, embeddingDim * 4);
        this.downProjection.initWeights(generator, embeddingDim * 4, embeddingDim);
    }

    @Override
    public void toDevice(DeviceType deviceType) {
        this.normalizer.toDevice(deviceType);
        this.upProjection.toDevice(deviceType);
        this.downProjection.toDevice(deviceType);
        this.attention.to(deviceType);
    }

    @Override
    public Tensor forward(ForwardContext context) {
        Tensor input = context.input();

        if (input.dimension() != 3) {
            throw new IllegalArgumentException(
                    "Input must have shape [batch_size, seq_len_dimension], got: " + Arrays.toString(input.shape())
            );
        }

        int[] shape = input.shape();

        int index = context.index();
        boolean training = context.training();

        int batchSize = shape[0];
        int seqLength = shape[1];
        int dimension = shape[2];

        StatesCache cache = context.cache();
        cache.setInput(index, input);

        Tensor result = Tensors.zeros(batchSize, seqLength, dimension);

        for (int b = 0; b < batchSize; b++) {
            Range range = new Range(b, b + 1);

            Tensor batch = input.slice(range).reshape(seqLength, dimension).withGrad();
            Tensor attended = attention.attend(cache, batch);

            if (training) {
                attended = dropout.forward(new ForwardContext(cache, attended, index, true));
            }

            Tensor added1 = batch.addGrad(attended);
            Tensor normalizedAttention = normalizer.forward(new ForwardContext(cache, added1, b, training));

            Tensor upProjected = upProjection.forward(new ForwardContext(cache, normalizedAttention, b, training));
            Tensor downProjected = downProjection.forward(new ForwardContext(cache, upProjected, b, training));

            if (training) {
                downProjected = dropout.forward(new ForwardContext(cache, downProjected, index, true));
            }

            Tensor added2 = normalizedAttention.addGrad(downProjected);
            Tensor normalized = normalizer.forward(new ForwardContext(cache, added2, b, training));

            float[] normalizedData = normalized.data();
            int offset = b * seqLength * dimension;

            System.arraycopy(normalizedData, 0, result.data(), offset, normalizedData.length);
        }

        cache.setOutput(index, result);
        return result;
    }

    @Override
    public int size() {
        return embeddingDim;
    }

    @Override
    public int totalWeights() {
        return upProjection.totalWeights() + downProjection.totalWeights() + normalizer.totalWeights() + attention.totalWeights();
    }
}
