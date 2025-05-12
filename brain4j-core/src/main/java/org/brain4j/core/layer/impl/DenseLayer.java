package org.brain4j.core.layer.impl;

import org.brain4j.core.clipping.GradientClipper;
import org.brain4j.core.clipping.impl.HardClipper;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.structure.StatesCache;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.tensor.Tensor;

/**
 * Represents a fully connected (dense) layer in a neural network.
 */
public class DenseLayer extends Layer {

    private GradientClipper clipper = new HardClipper(5);

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
    public String getLayerName() {
        return "Dense Layer";
    }

    @Override
    public Tensor forward(StatesCache cache, Tensor input, boolean training) {
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
            Z = layerNorm.forward(cache, Z, training);
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

        Tensor weightsNext = previous.getWeights();  // [n_out, n_out_next]
        Tensor deltaProjected = delta.matmul(weightsNext.transpose());  // [batch_size x n_out]

        Tensor deltaThisLayer = deltaProjected.mul(derivative); // [batch_size x n_out]

        Tensor gradient = optimizer.optimize(this, deltaThisLayer, input); // [n_in x n_out]
        Tensor biasGradient = deltaThisLayer.sum(0, false); // [n_out]

        clipper.clip(gradient);
        clipper.clip(biasGradient);

        updater.acknowledgeChange(this, gradient, biasGradient);
        return deltaThisLayer;
    }
}
