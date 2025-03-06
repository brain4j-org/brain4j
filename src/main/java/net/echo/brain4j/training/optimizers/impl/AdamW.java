package net.echo.brain4j.training.optimizers.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.updater.Updater;

import java.util.List;

public class AdamW extends Adam {

    private double weightDecay;

    public AdamW(double learningRate) {
        this(learningRate, 0.001);
    }

    public AdamW(double learningRate, double weightDecay) {
        this(learningRate, weightDecay, 0.9, 0.999, 1e-8);
    }

    public AdamW(double learningRate, double weightDecay, double beta1, double beta2, double epsilon) {
        super(learningRate, beta1, beta2, epsilon);
        this.weightDecay = weightDecay;
    }

    @Override
    public double update(StatesCache cache, Synapse synapse) {
        double adamValue = super.update(cache, synapse);
        double weightDecayTerm = weightDecay * synapse.getWeight();

        return adamValue + weightDecayTerm;
    }

    @Override
    public double update(StatesCache cache, int id, float gradient, float weight) {
        double adamValue = super.update(cache, id, gradient, weight);
        double weightDecayTerm = weightDecay * weight;

        return adamValue + weightDecayTerm;
    }

    @Override
    public void postIteration(StatesCache cacheHolder, Updater updater, List<Layer<?, ?>> layers) {
        this.timestep++;

        this.beta1Timestep = Math.pow(beta1, timestep);
        this.beta2Timestep = Math.pow(beta2, timestep);

        for (Layer<?, ?> layer : layers) {
            for (Synapse synapse : layer.getSynapses()) {
                float change = (float) update(cacheHolder, synapse);
                updater.acknowledgeChange(synapse, change);
            }
        }
    }

    public double getWeightDecay() {
        return weightDecay;
    }

    public void setWeightDecay(double weightDecay) {
        this.weightDecay = weightDecay;
    }
}