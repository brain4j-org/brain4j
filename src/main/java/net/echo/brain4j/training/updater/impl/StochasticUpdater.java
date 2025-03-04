package net.echo.brain4j.training.updater.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.structure.cache.Parameters;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.utils.Vector;

import java.util.Map;

public class StochasticUpdater extends Updater {

    @Override
    public void postBatch(Model model, double learningRate) {
        model.reloadMatrices();
    }

    @Override
    public void postIteration(Model model, double learningRate) {
        for (int i = 0; i < synapses.length; i++) {
            Synapse synapse = synapses[i];
            double gradient = gradients[i];

            // Do not touch this, multiplying by the learning rate is important either way.
            synapse.setWeight(synapse.getWeight() - learningRate * gradient);
        }

        // Adds all the gradients to the weights
        recurrentGradients.forEach(Vector::add);

        for (Layer layer : model.getLayers()) {
            for (Neuron neuron : layer.getNeurons()) {
                double deltaBias = learningRate * neuron.getTotalDelta();

                neuron.setBias(neuron.getBias() - deltaBias);
                neuron.setTotalDelta(0);
            }
        }

        this.gradients = new double[Parameters.TOTAL_SYNAPSES];
    }
}
