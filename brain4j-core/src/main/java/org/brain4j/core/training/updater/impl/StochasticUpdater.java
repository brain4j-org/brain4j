package org.brain4j.core.training.updater.impl;

import org.brain4j.core.model.Model;
import org.brain4j.core.training.updater.Updater;

public class StochasticUpdater extends Updater {

    @Override
    public void postBatch(Model model, double learningRate, int samples) {
        updateWeights(model, learningRate, samples);
        resetGradients(model);
    }
}