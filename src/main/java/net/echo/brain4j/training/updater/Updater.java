package net.echo.brain4j.training.updater;

import com.google.gson.annotations.JsonAdapter;
import net.echo.brain4j.adapters.UpdaterAdapter;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.structure.cache.Parameters;

@JsonAdapter(UpdaterAdapter.class)
public abstract class Updater {

    protected Synapse[] synapses;
    protected float[] gradients;

    public void acknowledgeChange(Synapse synapse, float change) {
        gradients[synapse.getSynapseId()] += change;
    }

    public void postInitialize(Sequential model) {
        this.synapses = new Synapse[Parameters.TOTAL_SYNAPSES];
        this.gradients = new float[Parameters.TOTAL_SYNAPSES];

        for (Layer<?, ?> layer : model.getLayers()) {
            for (Synapse synapse : layer.getSynapses()) {
                synapses[synapse.getSynapseId()] = synapse;
            }
        }
    }

    public void postIteration(Sequential model, double learningRate) {
    }

    public void postFit(Sequential model, double learningRate) {
    }

    public void postBatch(Sequential model, double learningRate) {
    }
}
