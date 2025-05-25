package org.brain4j.core.layer;

import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.core.training.StatesCache;

import java.util.List;
import java.util.Random;

/**
 * Abstract base class for all neural network layers.
 * Each layer processes input tensors, computes forward and backward passes independently,
 * and holds its own parameters such as weights, biases, activation function and gradient clipper.
 * @author xEcho1337
 */
public abstract class Layer {

    protected final Activation activation;
    protected final GradientClipper clipper;
    protected Tensor weights;
    protected Tensor bias;

    public Layer(Activation activation, GradientClipper clipper) {
        this.activation = activation;
        this.clipper = clipper;
    }

    /**
     * Performs a forward pass through this layer.
     * @param cache the cache of this inference
     * @param input the input tensor
     * @param index the index of this layer
     * @param training whether inference or training
     * @return the output tensor
     */
    public abstract Tensor forward(StatesCache cache, Tensor input, int index, boolean training);

    /**
     * Returns the output size of this layer, i.e. the number of neurons.
     * @return the output size
     */
    public abstract int size();

    public void connect(Layer previous) {
        // No-op
    }

    public void initWeights(Random generator, double bound) {
        // No-op
    }

    public void computeLoss(
        Updater updater,
        StatesCache cache,
        Tensor targets,
        Tensor outputs,
        LossFunction lossFunction,
        int index
    ) {
        Tensor error = outputs.minus(targets);
        Tensor derivatives = activation.getDerivative(outputs);

        Tensor output = cache.preActivation(index);
        Tensor delta = lossFunction.getDelta(error, derivatives);

        output.backward(delta);

        Tensor weightsGradient = weights.grad().transpose();
        Tensor biasesGradient = bias.grad().sum(0, false);

        updater.change(weightsGradient, biasesGradient, index);
    }

    public void backward(Updater updater, Optimizer optimizer, int index) {
        if (weights == null) return;

        Tensor weightsGradient = weights.grad().transpose();
        Tensor biasGradient = bias.grad().sum(0, false);

        weightsGradient = optimizer.step(index, this, weightsGradient);

        clipper.clip(weightsGradient);
        clipper.clip(biasGradient);

        updater.change(weightsGradient, biasGradient, index);
    }

    public boolean skipPropagate() {
        return false;
    }

    /**
     * Gets the activation function for this layer.
     * @return the activation function
     */
    public Activation activation() {
        return activation;
    }

    /**
     * Gets the weights of this layer.
     * @return the weights
     */
    public Tensor weights() {
        return weights;
    }

    /**
     * Gets the bias of this layer.
     * @return the bias
     */
    public Tensor bias() {
        return bias;
    }

    public int totalNeurons() {
        if (bias == null) return 0;

        return bias.elements();
    }

    public int totalWeights() {
        if (weights == null) return 0;

        return weights.elements();
    }
}
