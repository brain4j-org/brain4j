package net.echo.brain4j.training.updater;

import com.google.gson.annotations.JsonAdapter;
import net.echo.brain4j.adapters.json.UpdaterAdapter;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.Parameters;
import net.echo.math4j.math.tensor.Tensor;

@JsonAdapter(UpdaterAdapter.class)
public abstract class Updater {

    protected Tensor[] gradientsTensors;
    protected Tensor[] biasesTensors;

    protected void updateWeights(Model model, double learningRate, int samples) {
        if (model.getLayers().size() != gradientsTensors.length) {
            return; // TODO: Implement this for transformers
        }

        for (int i = 0; i < gradientsTensors.length; i++) {
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
        this.gradientsTensors = new Tensor[Parameters.TOTAL_LAYERS];
        this.biasesTensors = new Tensor[Parameters.TOTAL_LAYERS];
    }

    public void postFit(Model model, double learningRate, int samples) {
    }

    public void postBatch(Model model, double learningRate, int samples) {
    }
}
