package net.echo.brain4j.training.updater.impl;

import net.echo.brain4j.model.Model;
import net.echo.brain4j.training.updater.Updater;

public class NormalUpdater extends Updater {

    @Override
    public void postFit(Model model, double learningRate) {
        model.reloadWeights();
    }
}
