package net.echo.brain4j.training.updater;

import com.google.gson.annotations.JsonAdapter;
import net.echo.brain4j.adapters.UpdaterAdapter;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.structure.cache.Parameters;
import net.echo.brain4j.utils.Vector;

import java.util.HashMap;
import java.util.Map;

@JsonAdapter(UpdaterAdapter.class)
public abstract class Updater {

    protected Map<Vector, Vector> recurrentGradients;
    protected Synapse[] synapses;
    protected double[] gradients;

    public void acknowledgeChange(Synapse synapse, double change) {
        gradients[synapse.getSynapseId()] += change;
    }

    public void acknowledgeRecurrentChange(Vector recurrentWeights, Vector recurrentChanges) {
        recurrentGradients.put(recurrentWeights, recurrentChanges);
    }

    public void postInitialize(Model model) {
        this.synapses = new Synapse[Parameters.TOTAL_SYNAPSES];
        this.gradients = new double[Parameters.TOTAL_SYNAPSES];
        this.recurrentGradients = new HashMap<>();

        for (Layer layer : model.getLayers()) {
            for (Synapse synapse : layer.getSynapses()) {
                synapses[synapse.getSynapseId()] = synapse;
            }
        }
    }

    public void postIteration(Model model, double learningRate) {
    }

    public void postFit(Model model, double learningRate) {
    }

    public void postBatch(Model model, double learningRate) {
    }
}
