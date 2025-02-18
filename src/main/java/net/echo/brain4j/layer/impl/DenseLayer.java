package net.echo.brain4j.layer.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.training.data.DataSet;

/**
 * Represents a fully connected (dense) layer in a neural network.
 */
public class DenseLayer extends Layer {

    /**
     * Constructs a new DenseLayer instance.
     *
     * @param input the number of neurons (units) in this layer, which determines
     *              the layer's capacity to learn and represent data.
     * @param activation the activation function to be applied to the output
     *                   of each neuron, enabling non-linear transformations
     *                   of the input data.
     */
    public DenseLayer(int input, Activations activation) {
        super(input, activation);
    }
}