package net.echo.brain4j.layer.impl;

import net.echo.brain4j.clipping.GradientClipper;
import net.echo.brain4j.clipping.impl.HardClipper;
import net.echo.math.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.StatesCache;
import net.echo.math.BrainUtils;
import net.echo.math.tensor.Tensor;

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
     * @param activation the activation function to be applied to the output of each neuron
     */
    public DenseLayer(int input, Activations activation) {
        this(input, activation, new HardClipper(5));
    }

    public DenseLayer(int input, Activations activation, GradientClipper clipper) {
        super(input, activation);
        this.clipper = clipper;
    }

    @Override
    public Tensor forward(StatesCache cache, Layer lastLayer, Layer nextLayer, Tensor input, boolean training) {
        if (!(lastLayer instanceof DenseLayer denseLayer)) {
            throw new UnsupportedOperationException("Layer before must be a dense layer!");
        }

        int numNeurons = bias.elements();

        Tensor weights = denseLayer.getWeights().withGrad(); // last layer weights, which is [m, n]
        Tensor reshapedInput = input.reshape(input.elements(), 1).withGrad(); // [n, 1] matrix

        Tensor result = weights.withGrad()
                .matmulWithGrad(reshapedInput) // [m, n] x [n, 1] = [m, 1]
                .reshape(numNeurons)
                .withGrad()
                .addWithGrad(bias);

        if (nextLayer instanceof LayerNorm layerNorm) {
            result = layerNorm.forward(cache, this, null, result, training).withGrad();
        }

        Tensor activated = result.activateWithGrad(activation);
        cache.setOutputTensor(this, activated);

        return activated;
    }

    @Override
    public Tensor propagate(StatesCache cache, Layer previous, Tensor delta) {
        Tensor output = cache.getOutputTensor(this);
        Tensor derivative = activation.getDerivative(output);

        // delta as a matrix [n_out, 1]
        Tensor deltaMatrix = delta.reshape(delta.elements(), 1);
        Tensor transposedWeights = weights.transpose(); // weights as [n_in, n_out]

        Tensor newDelta = transposedWeights
                .matmul(deltaMatrix)
                .reshape(output.elements());

        Tensor deltaL = newDelta.mul(derivative);
        Tensor gradient = optimizer.optimize(this, deltaMatrix, output.transpose());

        Tensor clippedGradient = clipper.clip(gradient);
        Tensor clippedDelta = clipper.clip(deltaL);

        updater.acknowledgeChange(this, clippedGradient, clippedDelta);
        return deltaL;
    }
}
