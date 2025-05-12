package org.brain4j.core.layer.impl.transformers;

import org.brain4j.core.initialization.WeightInitializer;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.DropoutLayer;
import org.brain4j.core.layer.impl.LayerNorm;
import org.brain4j.core.structure.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.transformers.attention.MultiHeadAttention;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;
import org.brain4j.math.tensor.index.Range;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.Arrays;
import java.util.Random;

public class TrEncoder extends Layer {

    protected DenseLayer upProjection;
    protected DenseLayer downProjection;
    protected LayerNorm normalizer;
    protected DropoutLayer dropout;
    protected MultiHeadAttention attention;

    protected double dropoutRate = 0.1; // default value
    protected int heads;
    protected int embeddingDim;

    protected TrEncoder() {
        this.normalizer = new LayerNorm(0);
        this.dropout = new DropoutLayer(dropoutRate);
    }

    public TrEncoder(int numHeads, int embeddingDim, double dropoutRate) {
        this(numHeads, embeddingDim);
        this.dropoutRate = dropoutRate;
        this.dropout = new DropoutLayer(dropoutRate);
    }

    public TrEncoder(int numHeads, int embeddingDim) {
        super(Activations.LINEAR.getFunction());

        this.normalizer = new LayerNorm(embeddingDim);
        this.upProjection = new DenseLayer(embeddingDim * 4, Activations.RELU);
        this.downProjection = new DenseLayer(embeddingDim, Activations.LINEAR);
        
        this.heads = numHeads;
        this.embeddingDim = embeddingDim;
        this.dropout = new DropoutLayer(dropoutRate);
        this.attention = createAttention(numHeads, embeddingDim);
    }

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        throw new UnsupportedOperationException("Not implemented for " + this.getClass().getSimpleName());
    }

    @Override
    public void deserialize(DataInputStream stream) throws Exception {
        throw new UnsupportedOperationException("Not implemented for " + this.getClass().getSimpleName());
    }

    @Override
    public Tensor forward(
        int index, StatesCache cache,
        Tensor input,
        boolean training
    ) {
        int[] shape = input.shape();

        if (shape.length != 3) {
            throw new IllegalArgumentException(
                    "Input must have shape [batch_size, seq_len, dimension]! Got: " + Arrays.toString(shape)
            );
        }

        cache.setInputTensor(this, input);

        int batchSize = shape[0];
        int seqLength = shape[1];
        int dimension = shape[2];

        Tensor result = Tensors.create(batchSize, seqLength, dimension);

        for (int i = 0; i < batchSize; i++) {
            Tensor batch = input.slice(new Range(i, i + 1)).reshape(seqLength, dimension);

            Tensor attended = attention.attend(cache, batch);

            if (training) attended = dropout.forward(i, cache, attended, true);

            Tensor attentionOut = normalizer.forward(i, cache, batch.add(attended), training);
            Tensor cached = cache.getFeedForwardCache(i, this); // [tokens, dimension]

            if (cached == null) cached = Tensors.create(0, embeddingDim);

            int tokens = attentionOut.shape()[0];
            int cacheSize = cached.shape()[0];

            Range range = new Range(cacheSize, tokens);
            Tensor notCached = attentionOut.slice(range);

            Tensor upProjected = upProjection.forward(i, cache, notCached, training);
            Tensor downProjected = downProjection.forward(i, cache, upProjected, training);

            Tensor stacked = Tensors.stack(cached, downProjected);

            cache.setFeedForwardCache(i, this, stacked);

            if (training) stacked = dropout.forward(i, cache, stacked, true);

            stacked = normalizer.forward(i, cache, attentionOut.add(stacked), training);

            result.setChannel(i, stacked);
        }

        cache.setOutputTensor(this, result);
        return result;
    }

    @Override
    public Tensor backward(int index, StatesCache cache, Layer previous, Tensor delta) {
        return null;
    }

    @Override
    public void connect(Random generator, Layer previous, double bound) {
        this.attention.compile(generator, weightInit);

        DenseLayer dummyLayer = new DenseLayer(embeddingDim, Activations.LINEAR);

        this.downProjection.connect(generator, dummyLayer, bound);
        this.upProjection.connect(generator, downProjection, bound);
    }

    @Override
    public int getTotalWeights() {
        return getAttentionParams() + getMLPParams();
    }

    @Override
    public int getTotalNeurons() {
        return embeddingDim * 5;
    }

    @Override
    public void compile(WeightInitializer weightInit, Optimizer optimizer, Updater updater) {
        super.compile(weightInit, optimizer, updater);
        this.attention = createAttention(heads, embeddingDim);
    }

    public MultiHeadAttention createAttention(int heads, int embeddingDim) {
        return new MultiHeadAttention(heads, embeddingDim);
    }

    public MultiHeadAttention getAttention() {
        return attention;
    }

    public LayerNorm getNormalizer() {
        return normalizer;
    }

    public int getAttentionParams() {
        return getAttention().size();
    }

    public int getMLPParams() {
        return upProjection.getTotalWeights() + downProjection.getTotalWeights();
    }
}


