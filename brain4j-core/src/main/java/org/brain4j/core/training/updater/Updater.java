package org.brain4j.core.training.updater;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.model.Model;
import org.brain4j.math.tensor.Tensor;

public abstract class Updater {

    protected Tensor[] weightsGradients;
    protected Tensor[] biasesGradients;

    protected void updateWeights(Model model, double learningRate, int samples) {
        if (model.size() != weightsGradients.length) {
            return;
        }

        for (int i = 0; i < weightsGradients.length; i++) {
            Layer layer = model.layerAt(i);

            Tensor gradW = weightsGradients[i];
            Tensor biasW = biasesGradients[i];

            if (gradW != null) {
                layer.weights().sub(gradW.div(samples).mul(learningRate));
            }

            if (biasW != null) {
                layer.bias().sub(biasW.div(samples).mul(learningRate));
            }
        }
    }

    public void change(Tensor weightChange, Tensor biasChange, int index) {
        Tensor gradW = weightsGradients[index];
        Tensor biasW = biasesGradients[index];

        if (gradW == null) gradW = weightChange;
        else gradW.add(weightChange);

        if (biasW == null) biasW = biasChange;
        else biasW.add(biasChange);

        this.weightsGradients[index] = gradW;
        this.biasesGradients[index] = biasW;
    }

    public void resetGradients(Model model) {
        this.weightsGradients = new Tensor[model.size()];
        this.biasesGradients = new Tensor[model.size()];

        for (int i = 0; i < model.size(); i++) {
            Layer layer = model.layerAt(i);

            if (layer.weights() != null) {
                layer.weights().zerograd();
            }

            if (layer.bias() != null) {
                layer.bias().zerograd();
            }
        }
    }

    public void postFit(Model model, double learningRate, int samples) {
        // Nothing to see here
    }

    public void postBatch(Model model, double learningRate, int samples) {
        // Nothing to see here
    }
}