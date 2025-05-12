package org.brain4j.core.transformers.group;

import org.brain4j.core.initialization.WeightInitializer;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.transformers.TrEncoder;
import org.brain4j.core.structure.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public class EncoderGroup extends Layer {

    private final List<TrEncoder> layers;
    private final int groupSize;
    private final int numHeads;
    private final int dimension;

    public EncoderGroup(int groupSize, int numHeads, int dimension) {
        this.layers = new ArrayList<>();
        this.groupSize = groupSize;
        this.numHeads = numHeads;
        this.dimension = dimension;

        for (int i = 0; i < groupSize; i++) {
            layers.add(new TrEncoder(numHeads, dimension));
        }
    }

    @Override
    public void compile(
        WeightInitializer weightInit,
        Optimizer optimizer,
        Updater updater
    ) {
        for (TrEncoder layer : layers) {
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
            TrEncoder layer = layers.get(i - 1);
            nextDelta = layer.backward(index, cache, previous, nextDelta);
        }

        return nextDelta;
    }

    @Override
    public int getTotalWeights() {
        return layers.stream()
            .mapToInt(TrEncoder::getTotalWeights)
            .sum();
    }

    @Override
    public int getTotalNeurons() {
        return layers.stream()
            .mapToInt(TrEncoder::getTotalNeurons)
            .sum();
    }

    public List<TrEncoder> getLayers() {
        return layers;
    }

    public int getGroupSize() {
        return groupSize;
    }

    public int getNumHeads() {
        return numHeads;
    }

    public int getDimension() {
        return dimension;
    }
}
