package net.echo.brain4j.transformers.group;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.structure.StatesCache;
import net.echo.brain4j.training.optimizer.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.transformers.TransformerEncoder;
import net.echo.math4j.math.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class EncoderGroup extends Layer {

    private final List<TransformerEncoder> layers;
    private final int groupSize;
    private final int numHeads;
    private final int dimension;

    public EncoderGroup(int groupSize, int numHeads, int dimension) {
        this.layers = new ArrayList<>();
        this.groupSize = groupSize;
        this.numHeads = numHeads;
        this.dimension = dimension;

        for (int i = 0; i < groupSize; i++) {
            layers.add(new TransformerEncoder(numHeads, dimension));
        }
    }

    @Override
    public void init(Random generator) {
        for (TransformerEncoder layer : layers) {
            layer.init(generator);
        }
    }

    @Override
    public void compile(WeightInitializer weightInit, LossFunction lossFunction, Optimizer optimizer, Updater updater) {
        for (TransformerEncoder layer : layers) {
            layer.compile(weightInit, lossFunction, optimizer, updater);
        }
    }

    @Override
    public Tensor forward(StatesCache cache, Layer lastLayer, Tensor input, boolean training) {
        Tensor output = input;

        for (TransformerEncoder layer : layers) {
            output = layer.forward(cache, lastLayer, output, training);
        }

        return output;
    }

    @Override
    public Tensor propagate(StatesCache cache, Layer previous, Tensor delta) {
        Tensor nextDelta = delta;

        for (int i = layers.size(); i > 0; i--) {
            TransformerEncoder layer = layers.get(i - 1);
            nextDelta = layer.propagate(cache, previous, nextDelta);
        }

        return nextDelta;
    }

    @Override
    public int getTotalParams() {
        int total = 0;

        for (TransformerEncoder layer : layers) {
            total += layer.getTotalParams();
        }

        return total;
    }

    @Override
    public int getTotalNeurons() {
        int total = 0;

        for (TransformerEncoder layer : layers) {
            total += layer.getTotalNeurons();
        }

        return total;
    }

    public List<TransformerEncoder> getLayers() {
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
