package net.echo.brain4j.layer.impl.transformers;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.LayerNorm;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.initialization.WeightInitializer;
import net.echo.brain4j.structure.StatesCache;
import net.echo.brain4j.training.optimizer.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.transformers.attention.MultiHeadAttention;
import net.echo.brain4j.transformers.head.AttentionHead;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.TensorFactory;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.List;
import java.util.Random;

public class TrEncoder extends Layer {

    protected Sequential feedForward;
    protected LayerNorm normalizer;

    protected MultiHeadAttention attention;
    protected int heads;
    protected int embeddingDim;

    TrEncoder() {
        this.normalizer = new LayerNorm();
    }

    public TrEncoder(int numHeads, int embeddingDim) {
        super(Activations.LINEAR.getFunction());
        
        this.normalizer = new LayerNorm();
        this.feedForward = new Sequential(
                new DenseLayer(embeddingDim, Activations.LINEAR),
                new DenseLayer(4 * embeddingDim, Activations.RELU),
                new DenseLayer(embeddingDim, Activations.LINEAR)
        );

        this.heads = numHeads;
        this.embeddingDim = embeddingDim;
    }

    @Override
    public Tensor forward(StatesCache cache, Layer lastLayer, Layer nextLayer, Tensor input, boolean training) {
        cache.setInputTensor(this, input);

        Tensor attended = attention.attend(cache, input, training);
        Tensor normalized = normalizer.normalize(attended.add(input));

        List<Tensor> normAttention = TensorFactory.toList(normalized);
        List<Tensor> cached = cache.getFeedForwardForLayer(this);

        for (int i = 0; i < normAttention.size(); i++) {
            if (cached.size() <= i) {
                Tensor tensor = normAttention.get(i);
                Tensor output = feedForward.predict(tensor.reshape(embeddingDim));
                Tensor reshaped = output.reshape(1, embeddingDim);

                cached.add(reshaped);
            }
        }

        Tensor merged = TensorFactory.mergeTensors(cached);
        cache.setOutputTensor(this, merged);

        return normalizer.normalize(merged.add(normalized));
    }

    @Override
    public Tensor propagate(StatesCache cache, Layer previous, Tensor delta) {
        return null;
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

            Tensor Q = TensorFactory.zeros(0).deserialize(stream);
            Tensor K = TensorFactory.zeros(0).deserialize(stream);
            Tensor V = TensorFactory.zeros(0).deserialize(stream);

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
    public void init(Random generator) {
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
        int total = 0;

        for (Layer layer : feedForward.getLayers()) {
            total += layer.getTotalParams();
        }

        return total;
    }
}


