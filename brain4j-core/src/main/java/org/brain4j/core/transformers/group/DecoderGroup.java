package org.brain4j.core.transformers.group;

import org.brain4j.core.initialization.WeightInitializer;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.transformers.TrDecoder;
import org.brain4j.core.structure.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public class DecoderGroup extends Layer {

    private final List<TrDecoder> layers;
    private final int amount;
    private final int numHeads;
    private final int dimension;

    public DecoderGroup(
        int layersAmount,
        int numHeads,
        int dimension
    ) {
        this.layers = new ArrayList<>();
        this.amount = layersAmount;
        this.numHeads = numHeads;
        this.dimension = dimension;

        for (int i = 0; i < layersAmount; i++) {
            layers.add(new TrDecoder(numHeads, dimension));
        }
    }

    @Override
    public String getLayerName() {
        return "(%sx) Decoders".formatted(amount);
    }

    @Override
    public void compile(
        WeightInitializer weightInit,
        Optimizer optimizer,
        Updater updater
    ) {
        for (TrDecoder layer : layers) {
            layer.compile(weightInit, optimizer, updater);
        }
    }

    @Override
    public Tensor forward(
        int index,
        StatesCache cache,
        Tensor input,
        boolean training
    ) {
        Tensor output = input;

        for (int i = 0; i < layers.size(); i++) {
            output = layers.get(i).forward(i, cache, output, training);
        }

        return output;
    }

    @Override
    public Tensor backward(
        int index,
        StatesCache cache,
        Layer previous,
        Tensor delta
    ) {
        Tensor nextDelta = delta;

        for (int i = layers.size(); i > 0; i--) {
            TrDecoder layer = layers.get(i - 1);
            nextDelta = layer.backward(i, cache, previous, nextDelta);
        }

        return nextDelta;
    }

    @Override
    public int getTotalWeights() {
        int total = 0;

        for (TrDecoder layer : layers) {
            total += layer.getTotalWeights();
        }

        return total;
    }

    @Override
    public int getTotalNeurons() {
        int total = 0;

        for (TrDecoder layer : layers) {
            total += layer.getTotalNeurons();
        }

        return total;
    }

    public List<TrDecoder> getLayers() {
        return layers;
    }

    public int getAmount() {
        return amount;
    }

    public int getNumHeads() {
        return numHeads;
    }

    public int getDimension() {
        return dimension;
    }
}
