package net.echo.brain4j.training.updater.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.cache.Parameters;
import net.echo.brain4j.training.updater.Updater;
import net.echo.math4j.math.tensor.Tensor;

public class StochasticUpdater extends Updater {

    @Override
    public void postBatch(Model model, double learningRate) {
//        for (int i = 0; i < synapses.length; i++) {
//            Synapse synapse = synapses[i];
//            float gradient = gradients[i];
//
//            // Do not touch this, multiplying by the learning rate is important either way.
//            synapse.setWeight(synapse.getWeight() - learningRate * gradient);
//        }

//        for (Kernel kernel : kernels) {
//            Vector[] updates = kernel.getUpdates();
//
//            for (int j = 0; j < updates.length; j++) {
//                Vector update = updates[j];
//                Vector kernelValue = kernel.getValues()[j];
//
//                kernelValue.subtract(update.scale(learningRate));
//                kernel.getValues()[j] = kernelValue;
//            }
//
//            kernel.resetUpdates();
//        }
//
//        for (Layer<?, ?> layer : model.getLayers()) {
//            for (Neuron neuron : layer.getNeurons()) {
//                double deltaBias = learningRate * neuron.getTotalDelta();
//
//                neuron.setBias(neuron.getBias() - deltaBias);
//                neuron.setTotalDelta(0.0f);
//            }
//        }

        for (int i = 0; i < gradientsTensors.length; i++) {
            Layer<?, ?> layer = model.getLayers().get(i);

            Tensor gradW = gradientsTensors[i];
            Tensor biasW = biasesTensors[i];

            if (gradW != null) {
                layer.getWeights().sub(gradW);
            }

            if (biasW != null) {
                layer.getBias().sub(biasW);
            }
        }

        this.gradientsTensors = new Tensor[Parameters.TOTAL_LAYERS];
        this.biasesTensors = new Tensor[Parameters.TOTAL_LAYERS];

        model.reloadWeights();
    }
}
