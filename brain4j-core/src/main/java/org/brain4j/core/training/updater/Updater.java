package org.brain4j.core.training.updater;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.model.Model;
import org.brain4j.math.tensor.Tensor;

public abstract class Updater {

    protected Tensor[] gradientsTensors;
    protected Tensor[] biasesTensors;

    protected void updateWeights(Model model, double learningRate, int samples) {
        if (model.getLayers().size() != gradientsTensors.length) {
            return; // TODO: Implement this for transformers
        }

        for (int i = 1; i < gradientsTensors.length; i++) {
            Layer layer = model.getLayers().get(i);

            Tensor gradW = gradientsTensors[i];
            Tensor biasW = biasesTensors[i];

            if (gradW != null) {
                layer.getWeights().sub(gradW.div(samples).mul(learningRate));
            }

            if (biasW != null) {
                layer.getBias().sub(biasW.div(samples).mul(learningRate));
            }
        }
    }

    public void acknowledgeChange(Layer layer, Tensor change, Tensor biasDelta) {
        Tensor gradW = gradientsTensors[layer.getId()];
        Tensor biasW = biasesTensors[layer.getId()];

        if (gradW == null) gradW = change;
        else gradW.add(change);

        if (biasW == null) biasW = biasDelta;
        else biasW.add(biasDelta);

        this.gradientsTensors[layer.getId()] = gradW;
        this.biasesTensors[layer.getId()] = biasW;
    }

    public void postInitialize() {
        this.gradientsTensors = new Tensor[Layer.getTotalLayers()];
        this.biasesTensors = new Tensor[Layer.getTotalLayers()];
    }

    public void postFit(Model model, double learningRate, int samples) {
        postInitialize();
    }

    public void postBatch(Model model, double learningRate, int samples) {
    }
}
