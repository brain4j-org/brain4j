package net.echo.brain4j.training.updater.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.training.updater.Updater;

import java.util.List;

public class StochasticUpdater extends Updater {

    private Synapse[] synapses;
    private double[] gradients;

    @Override
    public void postInitialize() {
        this.synapses = new Synapse[Synapse.SYNAPSE_COUNTER];
        this.gradients = new double[Synapse.SYNAPSE_COUNTER];
    }

    @Override
    public void postIteration(List<Layer> layers, double learningRate) {
        for (int i = 0; i < synapses.length; i++) {
            Synapse synapse = synapses[i];
            double gradient = gradients[i];

            synapse.setWeight(synapse.getWeight() + learningRate * gradient);
        }

        this.gradients = new double[Synapse.SYNAPSE_COUNTER];

        for (Layer layer : layers) {
            for (Neuron neuron : layer.getNeurons()) {
                double deltaBias = learningRate * neuron.getTotalDelta();

                neuron.setBias(neuron.getBias() + deltaBias);
                neuron.setTotalDelta(0);
            }
        }
    }

    @Override
    public void acknowledgeChange(Synapse synapse, double change) {
        gradients[synapse.getSynapseId()] += change;
        synapses[synapse.getSynapseId()] = synapse;
    }
}
