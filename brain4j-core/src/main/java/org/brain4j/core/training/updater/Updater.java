package org.brain4j.core.training.updater;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.model.Model;
import org.brain4j.core.training.updater.impl.NormalUpdater;
import org.brain4j.core.training.updater.impl.StochasticUpdater;
import org.brain4j.math.tensor.Tensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Abstract class to define a gradient updater.
 * @see StochasticUpdater
 * @see NormalUpdater
 */
public abstract class Updater {

    protected Map<Layer, Tensor> weightsGradients;
    protected Map<Layer, Tensor> biasesGradients;

    protected void updateWeights(Model model, double learningRate, int samples) {
        model.updateWeights(layer -> {
            Tensor gradW = weightsGradients.get(layer);
            Tensor biasW = biasesGradients.get(layer);

            Tensor weights = layer.weights();
            Tensor biases = layer.bias();

            if (gradW != null && weights != null) {
                weights.sub(gradW.div(samples).mul(learningRate));
            }

            if (biasW != null && biases != null) {
                biases.sub(biasW.div(samples).mul(learningRate));
            }
        });
    }

    public void change(Tensor weightChange, Tensor biasChange, Layer layer) {
        Tensor gradW = weightsGradients.get(layer);
        Tensor biasW = biasesGradients.get(layer);

        if (weightChange != null) {
            if (gradW == null) gradW = weightChange;
            else gradW = gradW.add(weightChange);
        }

        if (biasChange != null) {
            if (biasW == null) biasW = biasChange;
            else biasW = biasW.add(biasChange);
        }

        weightsGradients.put(layer, gradW);
        biasesGradients.put(layer, biasW);
    }

    public void resetGradients(Model model) {
        this.weightsGradients = new HashMap<>();
        this.biasesGradients = new HashMap<>();

        model.zeroGrad();
    }

    public void postFit(Model model, double learningRate, int samples) {
        // Nothing to see here
    }

    public void postBatch(Model model, double learningRate, int samples) {
        // Nothing to see here
    }
}