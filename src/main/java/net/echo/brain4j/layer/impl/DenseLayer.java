package net.echo.brain4j.layer.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.convolution.Kernel;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.utils.Vector;

import java.util.List;

/**
 * Represents a fully connected (dense) layer in a neural network.
 */
public class DenseLayer extends Layer {

    protected Layer nextLayer;
    protected List<Vector> weights;

    /**
     * Constructs an instance of a dense layer.
     *
     * @param input the number of neurons in this layer
     * @param activation the activation function to be applied to the output of each neuron
     */
    public DenseLayer(int input, Activations activation) {
        super(input, activation);
    }

    @Override
    public Kernel forward(StatesCache cache, Layer lastLayer, Kernel input) {
        if (!(lastLayer instanceof DenseLayer denseLayer)) {
            throw new UnsupportedOperationException("Layer before must be a dense layer!");
        }

        List<Neuron> lastNeurons = lastLayer.getNeurons();

        int inSize = lastNeurons.size();
        int outSize = neurons.size();

        Vector inputVector = new Vector(inSize);

        for (int i = 0; i < inSize; i++) {
            inputVector.set(i, lastNeurons.get(i).getValue(cache));
        }

        for (int i = 0; i < outSize; i++) {
            double value = denseLayer.getWeights().get(i).weightedSum(inputVector);
            neurons.get(i).setValue(cache, value);
        }

        applyFunction(cache, lastLayer);
        return null;
    }

    public void updateWeights(Layer nextLayer, Vector[] weights) {
        this.nextLayer = nextLayer;
        this.weights = List.of(weights);
    }

    public List<Vector> getWeights() {
        return weights;
    }
}