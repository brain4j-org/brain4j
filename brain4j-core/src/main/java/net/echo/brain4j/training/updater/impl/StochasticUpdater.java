package net.echo.brain4j.training.updater.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.Parameters;
import net.echo.brain4j.training.updater.Updater;
import net.echo.math4j.math.tensor.Tensor;

import java.util.Arrays;

public class StochasticUpdater extends Updater {

    @Override
    public void postBatch(Model model, double learningRate) {
        System.out.println("called and " + gradientsTensors[0].sum());
        if (model.getLayers().size() != gradientsTensors.length) {
            return; // TODO: Implement this for transformers
        }

        for (int i = 0; i < gradientsTensors.length; i++) {
            Layer layer = model.getLayers().get(i);

            Tensor gradW = gradientsTensors[i];
            Tensor biasW = biasesTensors[i];

            if (gradW != null) {
                layer.getWeights().sub(gradW.mul(learningRate));
            }

            if (biasW != null) {
                layer.getBias().sub(biasW.mul(learningRate));
            }
        }
    }

    @Override
    public void postFit(Model model, double learningRate) {
        this.gradientsTensors = new Tensor[Parameters.TOTAL_LAYERS];
        this.biasesTensors = new Tensor[Parameters.TOTAL_LAYERS];
    }
}
