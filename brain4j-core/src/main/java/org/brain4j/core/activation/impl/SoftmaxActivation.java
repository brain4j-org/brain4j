package org.brain4j.core.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.weightsinit.WeightInitialization;
import org.brain4j.core.weightsinit.UniformXavierInit;

public class SoftmaxActivation implements Activation {

    private final double temperature;

    public SoftmaxActivation() {
        this(1.0);
    }

    public SoftmaxActivation(double temperature) {
        this.temperature = temperature;
    }

    @Override
    public WeightInitialization defaultWeightInit() {
        return new UniformXavierInit();
    }

    @Override
    public double activate(double input) {
        throw new UnsupportedOperationException("Softmax is a vector-based activation; use activate(Tensor) instead.");
    }

    @Override
    public Tensor activate(Tensor input) {
        return input.softmax(temperature);
    }

    @Override
    public double derivative(double input) {
        return input * (1.0 - input);
    }

    @Override
    public String kernelPrefix() {
        return "softmax";
    }
}
