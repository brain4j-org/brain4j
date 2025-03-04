package net.echo.brain4j.training.updater.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.recurrent.RecurrentLayer;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.structure.cache.Parameters;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.utils.Vector;

import java.util.List;

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

        for (Layer layer : model.getLayers()) {
            if (layer instanceof RecurrentLayer recurrentLayer) {
                List<Vector> recurrentWeights = recurrentLayer.getRecurrentWeights();

                for (int i = 0; i < recurrentWeights.size(); i++) {
                    Vector recurrentWeightVector = recurrentWeights.get(i);
                    for (int j = 0; j < recurrentWeightVector.size(); j++) {
                        int index = i * Parameters.TOTAL_NEURONS + j;

                        double gradient = recurrentGradients[index];
                        double newWeight = recurrentWeightVector.get(j) - learningRate * gradient;

                        recurrentWeightVector.set(j, newWeight);
                    }
                }
            }

            for (Neuron neuron : layer.getNeurons()) {
                double deltaBias = learningRate * neuron.getTotalDelta();

                neuron.setBias(neuron.getBias() - deltaBias);
                neuron.setTotalDelta(0);
            }
        }

        this.recurrentGradients = new double[Parameters.TOTAL_NEURONS * Parameters.TOTAL_NEURONS];
        this.gradients = new double[Parameters.TOTAL_SYNAPSES];
    }
}
