package org.brain4j.core.layer;

import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.core.clipper.impl.HardClipper;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.LinearActivation;
import org.brain4j.math.device.DeviceType;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.TensorGPU;
import org.brain4j.math.weights.WeightInitialization;

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

    public void connect(Layer previous) {
        // No-op
    }

    public void initWeights(Random generator, int input, int output) {
        // No-op
    }

    public void to(DeviceType deviceType) {
        if (this.weights != null)
            this.weights = weights.to(deviceType).withGrad();

        if (this.bias != null)
            this.bias = bias.to(deviceType).withGrad();
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

        Tensor delta = lossFunction.getDelta(error, derivatives);
        Tensor output = cache.preActivation(index);

//        System.out.println(error);
//        System.out.println(derivatives);
//        System.out.println(delta);
        output.backward(delta);

        Tensor weightsGrad = weights.grad().transpose();
        Tensor biasGrad = bias.grad().sum(0, false);

        updater.change(weightsGrad, biasGrad, index);
    }

    public void backward(Updater updater, Optimizer optimizer, int index) {
        if (weights == null) return;

        Tensor weightsGrad = weights.grad().transpose();
        Tensor biasGrad = bias.grad().sum(0, false);

        weightsGrad = optimizer.step(index, this, weightsGrad);

        clipper.clip(weightsGrad);
        clipper.clip(biasGrad);

        updater.change(weightsGrad, biasGrad, index);
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
