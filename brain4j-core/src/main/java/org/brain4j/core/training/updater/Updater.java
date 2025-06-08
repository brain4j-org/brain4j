package org.brain4j.core.training.updater;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.core.training.updater.impl.NormalUpdater;
import org.brain4j.core.training.updater.impl.StochasticUpdater;
import org.brain4j.math.tensor.Tensor;

/**
 * Abstract class to define a gradient updater.
 * @see StochasticUpdater
 * @see NormalUpdater
 */
public abstract class Updater {

    protected Tensor[] weightsGradients;
    protected Tensor[] biasesGradients;

    protected void updateWeights(Model model, double learningRate, int samples) {
        if (model.size() != weightsGradients.length) {
            return;
        }

        int index = 0;
        update(model, index, samples, learningRate);
    }

    private void update(Model model, int index, int samples, double learningRate) {
        for (Layer layer : model) {
            if (layer instanceof Model subModel) {
                update(subModel, index, samples, learningRate);
                continue;
            }

            Tensor gradW = weightsGradients[index];
            Tensor biasW = biasesGradients[index];

            Tensor weights = layer.weights();
            Tensor biases = layer.bias();

            if (gradW != null && weights != null) {
                weights.sub(gradW.div(samples).mul(learningRate));
            }

            if (biasW != null && biases != null) {
                biases.sub(biasW.div(samples).mul(learningRate));
            }

            index++;
        }
    }

    public void change(Tensor weightChange, Tensor biasChange, int index) {
        Tensor gradW = weightsGradients[index];
        Tensor biasW = biasesGradients[index];

        if (weightChange != null) {
            if (gradW == null) gradW = weightChange;
            else gradW = gradW.add(weightChange);
        }

        if (biasChange != null) {
            if (biasW == null) biasW = biasChange;
            else biasW = biasW.add(biasChange);
        }

        this.weightsGradients[index] = gradW;
        this.biasesGradients[index] = biasW;
    }

    public void resetGradients(Model model) {
        this.weightsGradients = new Tensor[model.size()];
        this.biasesGradients = new Tensor[model.size()];

        model.zeroGrad();
    }

    public void postFit(Model model, double learningRate, int samples) {
        // Nothing to see here
    }

    public void postBatch(Model model, double learningRate, int samples) {
        // Nothing to see here
    }
}