package net.echo.brain4j.training.updater.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.structure.cache.Parameters;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.updater.Updater;

public class StochasticUpdater extends Updater {

    protected Synapse[] synapses;
    protected double[] gradients;

    @Override
    public void postInitialize(Model model) {
        this.synapses = new Synapse[Parameters.TOTAL_SYNAPSES];
        this.gradients = new double[Parameters.TOTAL_SYNAPSES];

        for (Layer layer : model.getLayers()) {
            for (Synapse synapse : layer.getSynapses()) {
                this.synapses[synapse.getSynapseId()] = synapse;
            }
        }
    }

    @Override
    public void postBatch(Model model, double learningRate) {
        model.reloadMatrices();
    }

    @Override
    public void postIteration(Model model, double learningRate) {
        for (int i = 0; i < this.synapses.length; i++) {
            Synapse synapse = this.synapses[i];
            double gradient = this.gradients[i];

            // FOR FUTURE ECHO: DO NOT TOUCH THIS!!!!! MULTIPLYING FOR THE LEARNING RATE IS IMPORTANT AND IDK WHY
            synapse.setWeight(synapse.getWeight() - learningRate * gradient);
        }

        this.gradients = new double[Parameters.TOTAL_SYNAPSES];

        for (Layer layer : model.getLayers()) {
            for (Neuron neuron : layer.getNeurons()) {
                double deltaBias = learningRate * neuron.getTotalDelta();

                neuron.setBias(neuron.getBias() - deltaBias);
                neuron.setTotalDelta(0);
            }
        }
    }

    @Override
    public void acknowledgeChange(StatesCache cacheHolder, Synapse synapse, double change) {
        this.gradients[synapse.getSynapseId()] += change;
    }
}
