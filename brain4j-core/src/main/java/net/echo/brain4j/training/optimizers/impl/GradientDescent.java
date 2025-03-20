package net.echo.brain4j.training.optimizers.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.List;

public class GradientDescent extends Optimizer {

    public GradientDescent(double learningRate) {
        super(learningRate);
    }

    @Override
    public double update(StatesCache cache, Synapse synapse) {
        return learningRate * synapse.getOutputNeuron().getDelta(cache) * synapse.getInputNeuron().getValue(cache);
    }

    @Override
    public double update(StatesCache cache, int id, float gradient, float weight) {
        return learningRate * gradient;
    }

    @Override
    public void postIteration(StatesCache cacheHolder, Updater updater, List<Layer<?, ?>> layers) {
        for (Layer<?, ?> layer : layers) {
            for (Synapse synapse : layer.getSynapses()) {
                float change = (float) update(cacheHolder, synapse);
                updater.acknowledgeChange(synapse, change);
            }
        }
    }

    @Override
    public void serialize(DataOutputStream dataOutputStream) throws IOException {
        dataOutputStream.writeDouble(learningRate);
    }

    @Override
    public void deserialize(DataInputStream dataInputStream) throws IOException {
        this.learningRate = dataInputStream.readDouble();
    }
}
