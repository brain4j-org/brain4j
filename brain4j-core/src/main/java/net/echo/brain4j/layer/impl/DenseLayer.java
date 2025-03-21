package net.echo.brain4j.layer.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorCPU;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.math4j.math.vector.Vector;

/**
 * Represents a fully connected (dense) layer in a neural network.
 */
public class DenseLayer extends Layer<Vector, Vector> {

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
    public Vector forward(StatesCache cache, Layer<?, ?> lastLayer, Vector input) {
        if (!(lastLayer instanceof DenseLayer denseLayer)) {
            throw new UnsupportedOperationException("Layer before must be a dense layer!");
        }

        Tensor zTensor = denseLayer.getWeights().matmul(input);

        int numNeurons = zTensor.shape()[0];
        Vector output = new Vector(numNeurons);

        for (int i = 0; i < numNeurons; i++) {
            Neuron neuron = neurons.get(i);

            double z = zTensor.get(i, 0) + neuron.getBias();
            double activatedValue = activation.activate(z);

            neurons.get(i).setValue(cache, activatedValue);
            output.set(i, activatedValue);
        }

        return output;
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