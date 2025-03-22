package net.echo.brain4j.transformers;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.LayerNorm;
import net.echo.brain4j.loss.LossFunction;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.transformers.attention.MultiHeadAttention;

import java.util.ArrayList;
import java.util.List;

public class TransformerEncoder extends Layer<Tensor, Tensor> {

    private final Sequential feedForward;
    private final LayerNorm normalizer;

    private final int heads;
    private final int dimension;
    private final double temperature;

    private MultiHeadAttention attention;

    public TransformerEncoder(int numHeads, int dimension, double temperature) {
        super(0, Activations.LINEAR);

        this.normalizer = new LayerNorm();
        this.feedForward = new Sequential(
                new DenseLayer(dimension, Activations.LINEAR),
                new DenseLayer(4 * dimension, Activations.GELU),
                new DenseLayer(dimension, Activations.LINEAR)
        );

        this.heads = numHeads;
        this.dimension = dimension;
        this.temperature = temperature;
    }

    public int getAttentionSize() {
        return attention.getTotalNeurons();
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
    public void compile(WeightInitializer weightInit, LossFunction lossFunction, Optimizer optimizer, Updater updater) {
        this.attention = new MultiHeadAttention(weightInit, heads, dimension, temperature);
        this.feedForward.compile(weightInit, lossFunction, optimizer, updater);
    }

    @Override
    public Tensor forward(StatesCache cache, Layer<?, ?> lastLayer, Tensor input) {
        List<Tensor> inputTokens = tensorToList(input);
        
        List<Tensor> attentionOutput = attention.attendTensors(inputTokens);
        List<Tensor> normAttention = new ArrayList<>();

        for (Tensor token : attentionOutput) {
            normAttention.add(normalizer.normalize(token));
        }

        List<Tensor> feedForwardOutput = new ArrayList<>();

        for (Tensor tensor : normAttention) {
            Tensor output = feedForward.predict(tensor);
            feedForwardOutput.add(output.reshape(1, dimension));
        }

        List<Tensor> resultTokens = new ArrayList<>();

        for (int i = 0; i < feedForwardOutput.size(); i++) {
            Tensor tokenForwarded = feedForwardOutput.get(i);
            Tensor combined = tokenForwarded.add(normAttention.get(i));

            resultTokens.add(normalizer.normalize(combined));
        }

        return listToTensor(resultTokens);
    }

    private List<Tensor> tensorToList(Tensor input) {
        List<Tensor> tokens = new ArrayList<>();
        int sequenceLength = input.shape()[0];
        
        for (int i = 0; i < sequenceLength; i++) {
            Tensor token = input.slice(i, i+1).reshape(1, dimension);
            tokens.add(token);
        }
        
        return tokens;
    }
    
    private Tensor listToTensor(List<Tensor> tokens) {
        int sequenceLength = tokens.size();
        Tensor result = Tensor.zeros(sequenceLength, dimension);
        
        for (int i = 0; i < sequenceLength; i++) {
            Tensor token = tokens.get(i);
            for (int j = 0; j < dimension; j++) {
                result.set(token.get(0, j), i, j);
            }
        }
        
        return result;
    }

    public Sequential getFeedForward() {
        return feedForward;
    }

    public LayerNorm getNormalizer() {
        return normalizer;
    }

    public MultiHeadAttention getAttention() {
        return attention;
    }
}


