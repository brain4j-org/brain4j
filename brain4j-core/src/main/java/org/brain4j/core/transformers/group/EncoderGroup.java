package org.brain4j.core.transformers.group;

import org.brain4j.core.initialization.WeightInitializer;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.transformers.TrEncoder;
import org.brain4j.core.loss.LossFunction;
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
    public void compile(WeightInitializer weightInit, LossFunction lossFunction, Optimizer optimizer, Updater updater) {
        for (TrEncoder layer : layers) {
            layer.compile(weightInit, lossFunction, optimizer, updater);
        }
    }

    @Override
    public Tensor forward(StatesCache cache, Layer lastLayer, Tensor input, boolean training) {
        Tensor output = input;

        for (TrEncoder layer : layers) {
            output = layer.forward(cache, lastLayer, output, training);
        }

        return output;
    }

    @Override
    public Tensor backward(StatesCache cache, Layer previous, Tensor delta) {
        Tensor nextDelta = delta;

        for (int i = layers.size(); i > 0; i--) {
            TrEncoder layer = layers.get(i - 1);
            nextDelta = layer.backward(cache, previous, nextDelta);
        }

        return nextDelta;
    }

    @Override
    public int getTotalParams() {
        int total = 0;

        for (TrEncoder layer : layers) {
            total += layer.getTotalParams();
        }

        return total;
    }

    @Override
    public int getTotalNeurons() {
        int total = 0;

        for (TrEncoder layer : layers) {
            total += layer.getTotalNeurons();
        }

        return total;
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
