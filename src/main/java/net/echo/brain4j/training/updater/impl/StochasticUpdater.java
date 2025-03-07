package net.echo.brain4j.training.updater.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.structure.cache.Parameters;
import net.echo.brain4j.training.updater.Updater;

public class StochasticUpdater extends Updater {

    @Override
    public void postBatch(Sequential model, double learningRate) {
        for (int i = 0; i < synapses.length; i++) {
            Synapse synapse = synapses[i];
            float gradient = gradients[i];

            // Do not touch this, multiplying by the learning rate is important either way.
            synapse.setWeight(synapse.getWeight() - learningRate * gradient);
        }

        for (Layer<?, ?> layer : model.getLayers()) {
            for (Neuron neuron : layer.getNeurons()) {
                double deltaBias = learningRate * neuron.getTotalDelta();

                neuron.setBias(neuron.getBias() - deltaBias);
                neuron.setTotalDelta(0.0f);
            }
        }

        this.gradients = new float[Parameters.TOTAL_SYNAPSES];
        model.reloadMatrices();
    }
}
