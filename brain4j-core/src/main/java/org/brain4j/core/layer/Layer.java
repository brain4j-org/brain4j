package org.brain4j.core.layer;

import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.core.clipper.impl.HardClipper;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.activation.Activation;
import org.brain4j.core.activation.impl.LinearActivation;
import org.brain4j.math.device.DeviceType;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.weightsinit.WeightInitialization;

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

    /**
     * Constructs the tensors for weights in this layer.
     * @param previous the previous layer in the model
     * @return this layer by default
     */
    public Layer connect(Layer previous) {
        return this;
    }

    /**
     * Initializes the previously constructed weights with random values.
     * @param generator the random number generator
     * @param input the input dimension
     * @param output the output dimension
     */
    public void initWeights(Random generator, int input, int output) {
        // No-op
    }

    /**
     * Ports the weights of this layer to the specified device memory.
     * @param deviceType the device to port the weights on
     */
    public void toDevice(DeviceType deviceType) {
        if (weights != null) {
            weights = weights.to(deviceType).withGrad();
        }

        if (bias != null) {
            bias = bias.to(deviceType).withGrad();
        }
    }

    /**
     * Computes the loss (the gradient) with respect to the loss function and launches the autograd.
     * This method should only be called for the last layer of the neural network.
     *
     * @param cache the state cache of this inference
     * @param targets the target tensor
     * @param outputs the output tensor
     * @param lossFunction the loss function of this model
     */
    public void computeLoss(
        StatesCache cache,
        Tensor targets,
        Tensor outputs,
        LossFunction lossFunction
    ) {
        Tensor preOutput = cache.preActivation(this);

        Tensor error = outputs.minus(targets);
        Tensor derivatives = activation.derivative(preOutput);

        Tensor delta = lossFunction.getDelta(error, derivatives);
        preOutput.backward(delta);
    }

    /**
     * Computes the backward step for this layer, by calling the optimizer and scheduling weights update.
     * @param updater the updater of this model
     * @param optimizer the optimizer of this model
     * @param index the index of this layer
     */
    public void backward(Updater updater, Optimizer optimizer, int index) {
        if (weights == null) return;

        Tensor weightsGrad = weights.grad();
        Tensor biasGrad = bias.grad().sum(0, false);

        weightsGrad = optimizer.step(weights, weightsGrad);

        clipper.clip(weightsGrad);
        clipper.clip(biasGrad);

        updater.change(weights, weightsGrad);
        updater.change(bias, biasGrad);
    }

    /**
     * Validates if the input can be passed as an input to this layer.
     * This is done by checking the input dimension and comparing it
     * to the layer's expected dimension.
     *
     * @param input the input tensor
     * @return <code>true</code> if the input is valid, <code>false</code> otherwise
     */
    public boolean validateInput(Tensor input) {
        return true;
    }

    /**
     * Gets whether the {@linkplain Layer#backward} function should be called for this layer.
     * @return <code>true</code> if the layer should be skipped, <code>false</code> otherwise
     */
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
     * Gets the gradient clipping function for this layer.
     * @return the gradient clipping function
     */
    public GradientClipper clipper() {
        return clipper;
    }

    /**
     * Sets the gradient clipping function for this layer.
     * @param clipper the new gradient clipping function
     * @return this layer
     */
    public Layer clipper(GradientClipper clipper) {
        this.clipper = clipper;
        return this;
    }

    /**
     * Gets the weight initialization function for this layer.
     * @return the weight initialization function
     */
    public WeightInitialization weightInit() {
        return weightInit;
    }

    /**
     * Sets the weight initialization function for this layer.
     * @param weightInit the new weight initialization function
     * @return this layer
     */
    public Layer weightInit(WeightInitialization weightInit) {
        this.weightInit = weightInit;
        return this;
    }

    /**
     * Resets the gradients for all the weights in this layer.
     */
    public void resetGrad() {
        if (weights != null) {
            weights.zerograd();
        }

        if (bias != null) {
            bias.zerograd();
        }
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

    /**
     * Gets the total number of biases in this layer.
     * @return 0 if bias is <code>null</code>, otherwise the number of elements in the bias tensor
     */
    public int totalBiases() {
        if (bias == null) return 0;

        return bias.elements();
    }

    /**
     * Gets the total number of weights in this layer.
     * @return 0 if the weights is <code>null</code>, otherwise the number of elements in the weights tensor
     */
    public int totalWeights() {
        if (weights == null) return 0;

        return weights.elements();
    }
}
