package net.echo.brain4j.training.updater.impl;

import net.echo.brain4j.model.Model;
import net.echo.brain4j.training.updater.Updater;

public class StochasticUpdater extends Updater {

    @Override
    public void postBatch(Model model, double learningRate, int samples) {
        updateWeights(model, learningRate, samples);
        postInitialize();
    }
}
