package org.brain4j.core.training.updater.impl;

import org.brain4j.core.model.Model;
import org.brain4j.core.training.updater.Updater;

public class NormalUpdater extends Updater {

    @Override
    public void postFit(Model model, double learningRate, int samples) {
        updateWeights(model, learningRate, samples);
        resetGradients(model);
    }
}