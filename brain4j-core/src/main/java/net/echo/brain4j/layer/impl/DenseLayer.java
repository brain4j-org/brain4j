package net.echo.brain4j.layer.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;
import net.echo.math4j.math.vector.Vector;

/**
 * Represents a fully connected (dense) layer in a neural network.
 */
public class DenseLayer extends Layer<Tensor, Tensor> {

    private Tensor weights;

    private DenseLayer() {
    }

    /**
     * Constructs a dense layer instance.
     *
     * @param input the number of neurons in this layer
     * @param activation the activation function to be applied to the output of each neuron
     */
    public DenseLayer(int input, Activations activation) {
        super(input, activation);
    }

    @Override
    public Tensor forward(StatesCache cache, Layer<?, ?> lastLayer, Tensor input) {
        if (!(lastLayer instanceof DenseLayer denseLayer)) {
            throw new UnsupportedOperationException("Layer before must be a dense layer!");
        }

        // TODO: Fix overhead that is too high!
        int numNeurons = neurons.size();

        Tensor reshapedInput = input.reshape(input.elements(), 1);
        Tensor result = denseLayer.getWeights().matmul(reshapedInput).reshape(numNeurons);
        Tensor output = TensorFactory.create(numNeurons);

        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);
            output.set(result.get(i) + neuron.getBias(), i);
        }

        Tensor activated = activation.activate(output);

        for (int i = 0; i < activated.elements(); i++) {
            neurons.get(i).setValue(cache, activated.get(i));
        }

        return activated;
    }

    @Override
    public void propagate(StatesCache cache, Layer<?, ?> previous) {
        int nextLayerSize = nextLayer.getNeurons().size();

        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);

            double value = neuron.getValue(cache);
            double derivative = activation.getDerivative(value);

            for (int j = 0; j < nextLayerSize; j++) {
                Synapse synapse = synapses.get(i * nextLayerSize + j);

                float weightChange = calculateGradient(cache, synapse, derivative);
                updater.acknowledgeChange(synapse, weightChange);
            }
        }
    }

    @Override
    public void updateWeights(Tensor weights) {
        this.weights = weights;
    }

    public Tensor getWeights() {
        return weights;
    }
}