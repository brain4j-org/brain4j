package net.echo.brain4j.layer.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.math4j.BrainUtils;
import net.echo.math4j.math.tensor.Tensor;

/**
 * Represents a fully connected (dense) layer in a neural network.
 */
public class DenseLayer extends Layer<Tensor, Tensor> {

    private DenseLayer() {
    }

    /**
     * Constructs a dense layer instance.
     *
     * @param input      the number of neurons in this layer
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

        int numNeurons = bias.elements();

        Tensor reshapedInput = input.reshape(input.elements(), 1);
        Tensor result = denseLayer.getWeights()
                // .cpu() // TODO: Fix the overhead on GPU
                .matmul(reshapedInput)
                .reshape(numNeurons)
                .add(bias);

        cache.setOutputTensor(this, result);
        return activation.activate(result);
    }

    @Override
    public Tensor propagate(StatesCache cache, Layer<?, ?> previous, Tensor delta) {
        Tensor output = cache.getOutputTensor(this);
        Tensor derivative = activation.getDerivative(output);

        // delta as a matrix [n_out, 1]
        Tensor deltaMatrix = delta.reshape(delta.elements(), 1);
        Tensor transposedWeights = weights.transpose(); // weights as [n_in, n_out]

        Tensor newDelta = transposedWeights
                .matmul(deltaMatrix)
                .reshape(output.elements());

        // element-wise multiplication of delta and derivative
        Tensor deltaL = newDelta.mul(derivative).map(BrainUtils::clipGradient);
        Tensor gradient = optimizer.optimize(this, deltaMatrix, output.transpose())
                .map(BrainUtils::clipGradient);

        updater.acknowledgeChange(this, gradient, deltaL);
        return deltaL;
    }
}
