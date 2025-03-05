package net.echo.brain4j.layer.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.utils.Vector;

import java.util.List;

/**
 * Represents a fully connected (dense) layer in a neural network.
 */
public class DenseLayer extends Layer<Vector, Vector> {

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
    public Vector forward(StatesCache cache, Layer<?, ?> lastLayer, Vector input) {
        if (!(lastLayer instanceof DenseLayer denseLayer)) {
            throw new UnsupportedOperationException("Layer before must be a dense layer!");
        }

        int outSize = neurons.size();
        Vector output = new Vector(outSize);

        for (int i = 0; i < outSize; i++) {
            double value = denseLayer.getWeights().get(i).weightedSum(input);
            neurons.get(i).setValue(cache, value);
        }

        applyFunction(cache, lastLayer);

        for (int i = 0; i < size(); i++) {
            output.set(i, neurons.get(i).getValue(cache));
        }

        return output;
    }

    @Override
    public void updateWeights(Vector[] weights) {
        this.weights = List.of(weights);
    }

    public List<Vector> getWeights() {
        return weights;
    }
}