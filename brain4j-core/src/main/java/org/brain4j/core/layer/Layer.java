package org.brain4j.core.layer;

import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.core.clipper.impl.HardClipper;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.LinearActivation;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.weights.WeightInitialization;

import java.util.Arrays;
import java.util.Random;

/**
 * Abstract base class for all neural network layers.
 * Each layer processes input tensors, computes forward and backward passes independently,
 * and holds its own parameters such as weights, biases, activation function and gradient clipper.
 * @author xEcho1337
 */
public abstract class Layer {

    protected Activation activation = new LinearActivation();
    protected GradientClipper clipper = new HardClipper(5);
    protected WeightInitialization weightInit = activation.defaultWeightInit();

    protected Tensor weights;
    protected Tensor bias;
    protected Layer next;

    /**
     * Performs a forward pass through this layer.
     * @param context the forward context
     * @return the output tensor
     */
    public abstract Tensor forward(ForwardContext context);

    /**
     * Returns the output size of this layer, i.e. the number of neurons.
     * @return the output size
     */
    public abstract int size();

    public void connect(Layer previous, Layer next) {
        this.next = next;
    }

    public void initWeights(Random generator, int input, int output) {
        // No-op
    }

    public Tensor computeLoss(
        Updater updater,
        StatesCache cache,
        Tensor targets,
        Tensor outputs,
        LossFunction lossFunction,
        int index
    ) {
        Tensor error = outputs.minus(targets);
        Tensor derivatives = activation.getDerivative(outputs);

        Tensor delta = lossFunction.getDelta(error, derivatives); // [4, 1] -> [batch_size, output_size]

        Tensor output = cache.preActivation(index);
        output.backward(delta);

        Tensor autoWeightsGradient = weights.grad().transpose(); // OK
        Tensor autoBiasGradient = bias.grad().sum(0, false); // OK

//        Tensor input = cache.input(index);
//        Tensor weightsGradient = delta.transpose().matmul(input);
//        Tensor biasesGradient = delta.sum(0, false);

        updater.change(autoWeightsGradient, autoBiasGradient, index);

        return delta;
    }

    public Tensor backward(StatesCache cache, Updater updater, Optimizer optimizer, Layer previous, int index, Tensor delta) {
        if (weights == null) return delta;

//        Tensor input = cache.input(index);
//        Tensor output = cache.output(index);
//        Tensor derivative = activation.getDerivative(output); // [batch_size, n_out]
//
//        Tensor weightsNext = previous.weights(); // [n_out, n_out_next]
//        Tensor deltaProjected = delta.matmul(weightsNext); // [batch_size x n_out]
//
//        Tensor deltaThisLayer = deltaProjected.mul(derivative); // [batch_size x n_out]

//        Tensor weightsGradient = input.transpose().matmul(deltaThisLayer);
//        Tensor biasGradient = deltaThisLayer.sum(0, false);

        Tensor autoGradWeights = weights.grad();
        Tensor autoBiasGradient = bias.grad().sum(0, false);

        autoGradWeights = optimizer.step(index, this, autoGradWeights);

        clipper.clip(autoGradWeights);
        clipper.clip(autoBiasGradient);

        updater.change(autoGradWeights, autoBiasGradient, index);

        return null;
    }

    public boolean validateInput(Tensor input) {
        return true;
    }

    public boolean skipForward() {
        return false;
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

    public GradientClipper clipper() {
        return clipper;
    }

    public Layer clipper(GradientClipper clipper) {
        this.clipper = clipper;
        return this;
    }

    public WeightInitialization weightInit() {
        return weightInit;
    }

    public Layer weightInit(WeightInitialization weightInit) {
        this.weightInit = weightInit;
        return this;
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

    public int totalBiases() {
        if (bias == null) return 0;

        return bias.elements();
    }

    public int totalWeights() {
        if (weights == null) return 0;

        return weights.elements();
    }
}
