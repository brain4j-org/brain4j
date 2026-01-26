package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.weightsinit.impl.UniformXavierInit;
import org.brain4j.math.weightsinit.WeightInit;

public record SoftmaxActivation(double temperature) implements Activation {

    public SoftmaxActivation() {
        this(1.0);
    }

    @Override
    public WeightInit defaultWeightInit() {
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
    public String getKernelPrefix() {
        return "softmax";
    }
}
