package net.echo.brain4j.training.updater.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.Parameters;
import net.echo.brain4j.training.updater.Updater;
import net.echo.math4j.math.tensor.Tensor;

public class NormalUpdater extends Updater {

    @Override
    public void postFit(Model model, double learningRate) {
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

        System.out.println("RESETTING!");
        this.gradientsTensors = new Tensor[Parameters.TOTAL_LAYERS];
        this.biasesTensors = new Tensor[Parameters.TOTAL_LAYERS];
    }
}
