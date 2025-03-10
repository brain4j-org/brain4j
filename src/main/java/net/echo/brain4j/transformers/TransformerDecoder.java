package net.echo.brain4j.transformers;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.LayerNorm;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.transformers.attention.MultiHeadAttention;
import net.echo.brain4j.transformers.masked.MaskedMultiHeadAttention;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.List;

public class TransformerDecoder extends Layer<List<Vector>, List<Vector>> {

    private final int heads;
    private final int dimension;
    private final double temperature;

    private final Sequential feedForward;
    private final LayerNorm normalizer;

    private MaskedMultiHeadAttention maskedAttention;

    public TransformerDecoder(int numHeads, int dimension, double temperature) {
        super(0, Activations.LINEAR);
        this.heads = numHeads;
        this.dimension = dimension;
        this.temperature = temperature;

        this.normalizer = new LayerNorm();
        this.feedForward = new Sequential(
                new DenseLayer(dimension, Activations.LINEAR),
                new DenseLayer(4 * dimension, Activations.GELU),
                new DenseLayer(dimension, Activations.LINEAR)
        );
    }

    public int getAttentionSize() {
        return maskedAttention.getTotalNeurons();
    }

    public int getFeedForwardSize() {
        return feedForward.getTotalWeights();
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
    public void compile(WeightInit weightInit, LossFunctions lossFunction, Optimizer optimizer, Updater updater) {
        this.maskedAttention = new MaskedMultiHeadAttention(weightInit, heads, dimension, temperature);
        this.feedForward.compile(weightInit, lossFunction, optimizer, updater);
    }

    @Override
    public void propagate(StatesCache cache, Layer<?, ?> previous) {
//        feedForward.propagate(cache, this, updater, optimizer);
//        maskedAttention.propagate(cache, this, updater, optimizer);
    }

    @Override
    public List<Vector> forward(StatesCache cache, Layer<?, ?> lastLayer, List<Vector> input) {
        List<Vector> attentionOutput = maskedAttention.attend(input);
        List<Vector> normAttention = new ArrayList<>();

        for (Vector token : attentionOutput) {
            normAttention.add(normalizer.normalize(token));
        }

        List<Vector> feedForwardOutput = new ArrayList<>();

        for (Vector vector : normAttention) {
            feedForwardOutput.add(feedForward.predict(vector));
        }

        List<Vector> result = new ArrayList<>();

        for (int i = 0; i < feedForwardOutput.size(); i++) {
            Vector tokenFF = feedForwardOutput.get(i);

            tokenFF.add(normAttention.get(i));
            result.add(normalizer.normalize(tokenFF));
        }

        return result;
    }

    public Sequential getFeedForward() {
        return feedForward;
    }

    public LayerNorm getNormalizer() {
        return normalizer;
    }

    public MultiHeadAttention getMaskedAttention() {
        return maskedAttention;
    }
}
