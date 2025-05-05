package org.brain4j.core.layer.impl.transformers;

import org.brain4j.core.initialization.WeightInitializer;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.layer.impl.DropoutLayer;
import org.brain4j.core.layer.impl.LayerNorm;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.core.structure.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.transformers.attention.MultiHeadAttention;
import org.brain4j.core.transformers.head.AttentionHead;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.List;
import java.util.Random;

public class TrEncoder extends Layer {

    protected Sequential feedForward;
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
        this.feedForward = new Sequential(
                new DenseLayer(embeddingDim, Activations.LINEAR),
                new DenseLayer(4 * embeddingDim, Activations.RELU),
                new DenseLayer(embeddingDim, Activations.LINEAR)
        );

        this.heads = numHeads;
        this.embeddingDim = embeddingDim;
        this.dropout = new DropoutLayer(dropoutRate);
        this.attention = createAttention(numHeads, embeddingDim);
    }

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        super.serialize(stream);
        stream.writeInt(embeddingDim);
        stream.writeInt(heads);

        for (AttentionHead head : attention.getHeads()) {
            Tensor Q = head.getQueryWeightsTensor();
            Tensor K = head.getKeyWeightsTensor();
            Tensor V = head.getValueWeightsTensor();

            Q.serialize(stream);
            K.serialize(stream);
            V.serialize(stream);
        }

        feedForward.serialize(stream);
    }

    @Override
    public void deserialize(DataInputStream stream) throws Exception {
        super.deserialize(stream);
        this.embeddingDim = stream.readInt();
        this.heads = stream.readInt();
        this.attention = createAttention(heads, embeddingDim);

        this.feedForward = new Sequential(
                new DenseLayer(embeddingDim, Activations.LINEAR),
                new DenseLayer(4 * embeddingDim, Activations.RELU),
                new DenseLayer(embeddingDim, Activations.LINEAR)
        );

        for (int i = 0; i < heads; i++) {
            AttentionHead head = attention.getHeads().get(i);

            Tensor Q = Tensors.zeros(0).deserialize(stream);
            Tensor K = Tensors.zeros(0).deserialize(stream);
            Tensor V = Tensors.zeros(0).deserialize(stream);

            head.setQueryWeightsTensor(Q);
            head.setKeyWeightsTensor(K);
            head.setValueWeightsTensor(V);
        }

        feedForward.deserialize(stream);

        for (Layer layer : feedForward.getLayers()) {
            layer.compile(weightInit, lossFunction, optimizer, updater);
        }
    }

    @Override
    public Tensor forward(StatesCache cache, Layer lastLayer, Tensor input, boolean training) {
        cache.setInputTensor(this, input);

        Tensor attended = attention.attend(cache, input);

        if (training) attended = dropout.forward(cache, this, attended, true);

        Tensor attentionOut = normalizer.normalize(input.add(attended));

        List<Tensor> normAttention = Tensors.toList(attentionOut);
        List<Tensor> cached = cache.getFeedForwardForLayer(this);

        for (int i = 0; i < normAttention.size(); i++) {
            if (cached.size() <= i) {
                Tensor tensor = normAttention.get(i);
                Tensor output = feedForward.predict(tensor.vector());
                Tensor reshaped = output.reshape(1, embeddingDim);

                cached.add(reshaped);
            }
        }

        Tensor merged = Tensors.mergeTensors(cached);
        cache.setOutputTensor(this, merged);

        if (training) merged = dropout.forward(cache, this, merged, true);

        return normalizer.normalize(attentionOut.add(merged));
    }

    @Override
    public Tensor backward(StatesCache cache, Layer previous, Tensor delta) {
        return null;
    }

    @Override
    public void connect(Random generator, Layer previous, double bound) {
        this.attention.compile(generator, weightInit);
    }

    @Override
    public int getTotalParams() {
        return getAttentionSize() + getFeedForwardSize();
    }

    @Override
    public int getTotalNeurons() {
        return feedForward.getTotalNeurons();
    }

    @Override
    public void compile(WeightInitializer weightInit, LossFunction lossFunction, Optimizer optimizer, Updater updater) {
        super.compile(weightInit, lossFunction, optimizer, updater);
        this.attention = createAttention(heads, embeddingDim);
        this.feedForward.compile(weightInit, lossFunction, optimizer, updater);
    }

    public MultiHeadAttention createAttention(int heads, int embeddingDim) {
        return new MultiHeadAttention(heads, embeddingDim);
    }

    public MultiHeadAttention getAttention() {
        return attention;
    }

    public Sequential getFeedForward() {
        return feedForward;
    }

    public LayerNorm getNormalizer() {
        return normalizer;
    }

    public int getAttentionSize() {
        return getAttention().getTotalNeurons();
    }

    public int getFeedForwardSize() {
        return feedForward.getLayers()
                .stream()
                .mapToInt(Layer::getTotalParams)
                .sum();
    }
}


