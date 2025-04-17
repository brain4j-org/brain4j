package net.echo.brain4j.training.updater.impl;

import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.Parameters;
import net.echo.brain4j.training.updater.Updater;
import net.echo.math.tensor.Tensor;

public class NormalUpdater extends Updater {

    @Override
    public void postFit(Model model, double learningRate, int samples) {
        updateWeights(model, learningRate, samples);
        this.gradientsTensors = new Tensor[Parameters.TOTAL_LAYERS];
        this.biasesTensors = new Tensor[Parameters.TOTAL_LAYERS];
    }
}
