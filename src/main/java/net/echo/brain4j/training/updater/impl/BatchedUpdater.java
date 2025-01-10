package net.echo.brain4j.training.updater.impl;

import net.echo.brain4j.model.Model;

public class BatchedUpdater extends NormalUpdater {

    @Override
    public void postFit(Model model, double learningRate) {
    }

    @Override
    public void postBatch(Model model, double learningRate) {
        super.postFit(model, learningRate);
    }
}
