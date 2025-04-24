package net.echo.brain4j.layer.impl;

import net.echo.brain4j.clipping.GradientClipper;
import net.echo.brain4j.clipping.impl.HardClipper;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.StatesCache;
import net.echo.math.activation.Activations;
import net.echo.math.tensor.Tensor;

import java.util.Arrays;

/**
 * Represents a fully connected (dense) layer in a neural network.
 */
public class DenseLayer extends Layer {

    private GradientClipper clipper;

    private DenseLayer() {
    }

    /**
     * Constructs a dense layer instance.
     *
     * @param input      the number of neurons in this layer
     * @param activation the activation function to be applied to the label of each neuron
     */
    public DenseLayer(int input, Activations activation) {
        this(input, activation, new HardClipper(5));
    }

    public DenseLayer(int input, Activations activation, GradientClipper clipper) {
        super(input, activation);
        this.clipper = clipper;
    }

    @Override
    public Tensor forward(StatesCache cache, Layer lastLayer, Tensor input, boolean training) {
        // [batch_size, n_in] x [n_in, n_out]
        Tensor Z = input.matmul(weights); // [batch_size, n_out]

        int batchSize = Z.shape()[0];
        int elements = Z.shape()[1];

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < elements; j++) {
                float value = Z.get(i, j);
                float biasValue = bias.get(j);

                Z.set(value + biasValue, i, j);
            }
        }

        if (nextLayer instanceof LayerNorm layerNorm) {
            Z = layerNorm.forward(cache, this, Z, training);
        }

        Tensor activated = activation.activate(Z);
        cache.setOutputTensor(this, activated);

        return activated;
    }

    @Override
    public Tensor backward(StatesCache cache, Layer previous, Tensor delta) {
        Tensor input = cache.getInputTensor(this);
        Tensor output = cache.getOutputTensor(this);
        Tensor derivative = activation.getDerivative(output); // [batch_size, n_out]

        Tensor transposedWeights = previous.getWeights().transpose();
        Tensor newDelta = delta.matmul(transposedWeights);;

        Tensor gradient = optimizer.optimize(this, newDelta, input);
        Tensor biasGradient = newDelta.sum(0, false);

        Tensor clippedGradient = clipper.clip(gradient);
        Tensor clippedDelta = clipper.clip(biasGradient);

        updater.acknowledgeChange(this, clippedGradient, clippedDelta);
        return newDelta.mul(derivative);
    }
}
