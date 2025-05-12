package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.tensor.Tensor;

public class SoftmaxActivation implements Activation {

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
