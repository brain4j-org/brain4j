package net.echo.brain4j.training.optimizers;

import com.google.gson.annotations.JsonAdapter;
import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.adapters.OptimizerAdapter;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.threading.NeuronCacheHolder;
import net.echo.brain4j.training.updater.Updater;

import java.util.List;

/**
 * Interface providing various methods to optimize training.
 */
@JsonAdapter(OptimizerAdapter.class)
public abstract class Optimizer {

    private static final double GRADIENT_CLIP = 10.0;
    protected double learningRate;

    /**
     * Initializes the optimizer with a specified learning rate.
     *
     * @param learningRate the learning rate
     */
    public Optimizer(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * Updates the given synapse based on the optimization algorithm.
     *
     * @param synapse the synapse to update
     */
    public abstract double update(NeuronCacheHolder cacheHolder, Synapse synapse, Object... params);

    /**
     * Called after the network has been compiled and all the synapses have been initialized.
     */
    public void postInitialize(Model model) {
    }

    /**
     * Gets the current learning rate.
     *
     * @return learning rate
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * Sets a new learning rate.
     *
     * @param learningRate the new learning rate
     */
    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    /**
     * Called after a sample has been iterated.
     *
     * @param updater the backpropagation updater
     * @param layers the layers of the model
     */
    public void postIteration(NeuronCacheHolder cacheHolder, Updater updater, List<Layer> layers) {
    }

    /**
     * Updates the given synapse based on the optimization algorithm.
     *
     * @param cacheHolder holds the neuron values for the current thread
     * @param layer       the layer of the neuron
     * @param neuron      the input neuron connected to the synapse
     * @param synapse     the synapse involved
     */
    public void applyGradientStep(NeuronCacheHolder cacheHolder, Updater updater, Layer layer, Neuron neuron, Synapse synapse) {
        double weightChange = calculateGradient(cacheHolder, layer, neuron, synapse);
        updater.acknowledgeChange(synapse, weightChange);
    }

    /**
     * Calculate the gradient for a synapse based on the delta and the value of the input.
     *
     * @param cacheHolder the cache holder for neuron values
     * @param neuron      the neuron
     * @param synapse     the synapse
     *
     * @return the calculated gradient
     */
    public double calculateGradient(NeuronCacheHolder cacheHolder, Layer layer, Neuron neuron, Synapse synapse) {
        double output = neuron.getValue(cacheHolder);

        Activation activationFunction = layer.getActivation().getFunction();

        double derivative;
        try {
            derivative = activationFunction.getDerivative(output);
        } catch (UnsupportedOperationException e) {

            List<Neuron> layerNeurons = layer.getNeurons();
            double[] layerOutputs = new double[layerNeurons.size()];

            for (int i = 0; i < layerNeurons.size(); i++) {
                layerOutputs[i] = layerNeurons.get(i).getValue(cacheHolder);
            }

            double[][] jacobian = activationFunction.getDerivativeMatrix(layerOutputs);

            int neuronIndex = layerNeurons.indexOf(neuron);
            if (neuronIndex < 0) {
                throw new IllegalStateException("Neuron not found in layer!");
            }

            derivative = jacobian[neuronIndex][neuronIndex];
        } catch (Exception e) {
            throw new RuntimeException("Failed to calculate gradient for neuron " + neuron.getId(), e);
        }

        double error = clipGradient(
                synapse.getWeight() * synapse.getOutputNeuron().getDelta(cacheHolder)
        );

        double delta = clipGradient(error * derivative);

        neuron.setDelta(cacheHolder, delta);

        double inputVal = synapse.getInputNeuron().getValue(cacheHolder);
        return clipGradient(delta * inputVal);
    }

    /**
     * Clips the gradient to avoid gradient explosion.
     *
     * @param gradient the gradient
     * @return the clipped gradient
     */
    public double clipGradient(double gradient) {
        return Math.max(Math.min(gradient, GRADIENT_CLIP), -GRADIENT_CLIP);
    }
}