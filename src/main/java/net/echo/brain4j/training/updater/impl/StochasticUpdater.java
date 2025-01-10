package net.echo.brain4j.training.updater.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.training.updater.Updater;

import java.util.List;

public class StochasticUpdater extends Updater {

    private Synapse[] synapses;
    private double[] gradients;

    @Override
    public void postInitialize(Model model) {
        this.synapses = new Synapse[Synapse.SYNAPSE_COUNTER];
        this.gradients = new double[Synapse.SYNAPSE_COUNTER];

        for (Layer layer : model.getLayers()) {
            for (Synapse synapse : layer.getSynapses()) {
                synapses[synapse.getSynapseId()] = synapse;
            }
        }
    }

    @Override
    public void postIteration(Model model, double learningRate) {
        for (int i = 0; i < synapses.length; i++) {
            Synapse synapse = synapses[i];
            double gradient = gradients[i];

            synapse.setWeight(synapse.getWeight() + learningRate * gradient);
        }

        model.reloadMatrices();

        this.gradients = new double[Synapse.SYNAPSE_COUNTER];

        for (Layer layer : model.getLayers()) {
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
    }
}
