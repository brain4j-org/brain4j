package org.brain4j.core.training.updater;

import org.brain4j.core.model.Model;
import org.brain4j.core.training.updater.impl.NormalUpdater;
import org.brain4j.core.training.updater.impl.StochasticUpdater;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.util.HashMap;
import java.util.Map;

/**
 * Abstract class to define a gradient updater.
 * @see StochasticUpdater
 * @see NormalUpdater
 */
public abstract class Updater {

    protected Map<Tensor, Tensor> weightsGradients = new HashMap<>();

    protected void updateWeights(double learningRate, int samples) {
        for (Map.Entry<Tensor, Tensor> entry : weightsGradients.entrySet()) {
            Tensor weights = entry.getKey();
            Tensor gradient = entry.getValue();

            if (gradient != null && weights != null) {
                weights.sub(gradient.div(samples).mul(learningRate));
            }
        }
    }

    public void change(Tensor weights, Tensor gradient) {
        weightsGradients.compute(weights, (w, g) -> {
            if (g == null) {
                return gradient.clone();
            }

            return g.add(gradient);
        });
    }

    public void resetGradients(Model model) {
        this.weightsGradients.clear();
        model.zeroGrad();
    }

    public void postFit(Model model, double learningRate, int samples) {
        // Nothing to see here
    }

    public void postBatch(Model model, double learningRate, int samples) {
        // Nothing to see here
    }
}