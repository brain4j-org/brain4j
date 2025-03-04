package net.echo.brain4j.training.optimizers.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;

import java.util.List;

public class GradientDescent extends Optimizer {

    public GradientDescent(double learningRate) {
        super(learningRate);
    }

    @Override
    public double update(StatesCache cacheHolder, Synapse synapse) {
        return learningRate * synapse.getOutputNeuron().getDelta(cacheHolder) * synapse.getInputNeuron().getValue(cacheHolder);
    }

    @Override
    public void postIteration(StatesCache cacheHolder, Updater updater, List<Layer> layers) {
        for (Layer layer : layers) {
            for (Synapse synapse : layer.getSynapses()) {
                double change = update(cacheHolder, synapse);
                updater.acknowledgeChange(synapse, change);
            }
        }
    }
}
