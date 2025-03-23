package net.echo.brain4j.training.updater;

import com.google.gson.annotations.JsonAdapter;
import net.echo.brain4j.adapters.json.UpdaterAdapter;
import net.echo.brain4j.convolution.Kernel;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.convolution.ConvLayer;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.structure.cache.Parameters;

import java.util.List;

@JsonAdapter(UpdaterAdapter.class)
public abstract class Updater {

    protected Kernel[] kernels;
    protected Synapse[] synapses;
    protected float[] gradients;

    public void acknowledgeChange(Synapse synapse, float change) {
        gradients[synapse.getSynapseId()] += change;
    }

    public void postInitialize(Sequential model) {
        this.kernels = new Kernel[Parameters.TOTAL_KERNELS];
        this.synapses = new Synapse[Parameters.TOTAL_SYNAPSES];
        this.gradients = new float[Parameters.TOTAL_SYNAPSES];

        for (Layer<?, ?> layer : model.getLayers()) {
            for (Synapse synapse : layer.getSynapses()) {
                synapses[synapse.getSynapseId()] = synapse;
            }

            if (layer instanceof ConvLayer convLayer) {
                List<Kernel> filters = convLayer.getKernels();

                for (Kernel filter : filters) {
                    kernels[filter.getId()] = filter;
                }
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
