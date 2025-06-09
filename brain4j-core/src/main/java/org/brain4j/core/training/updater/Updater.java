package org.brain4j.core.training.updater;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.model.Model;
import org.brain4j.core.training.updater.impl.NormalUpdater;
import org.brain4j.core.training.updater.impl.StochasticUpdater;
import org.brain4j.math.tensor.Tensor;

import java.util.List;

/**
 * Abstract class to define a gradient updater.
 * @see StochasticUpdater
 * @see NormalUpdater
 */
public abstract class Updater {

    protected Tensor[] weightsGradients;
    protected Tensor[] biasesGradients;

    protected void updateWeights(Model model, double learningRate, int samples) {
        int size = model.flattened().size();

        if (size != weightsGradients.length) {
            return;
        }

        List<Layer> flattened = model.flattened();

        for (int i = 0; i < flattened.size(); i++) {
            Layer layer = flattened.get(i);

            Tensor gradW = weightsGradients[i];
            Tensor biasW = biasesGradients[i];

            Tensor weights = layer.weights();
            Tensor biases = layer.bias();

            if (gradW != null && weights != null) {
                weights.sub(gradW.div(samples).mul(learningRate));
            }

            if (biasW != null && biases != null) {
                biases.sub(biasW.div(samples).mul(learningRate));
            }
        }
    }

    private void update(Model model, int index, int samples, double learningRate) {

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
        int size = model.flattened().size();

        this.weightsGradients = new Tensor[size];
        this.biasesGradients = new Tensor[size];

        model.zeroGrad();
    }

    public void postFit(Model model, double learningRate, int samples) {
        // Nothing to see here
    }

    public void postBatch(Model model, double learningRate, int samples) {
        // Nothing to see here
    }
}