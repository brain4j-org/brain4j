package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.weights.WeightInitialization;
import org.brain4j.math.weights.impl.UniformXavierInit;

public class SoftmaxActivation implements Activation {

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
        return input.softmax();
    }

    @Override
    public double getDerivative(double input) {
        return input * (1.0 - input);
    }
}
