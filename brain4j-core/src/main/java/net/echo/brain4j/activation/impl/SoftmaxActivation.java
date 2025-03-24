package net.echo.brain4j.activation.impl;

import net.echo.brain4j.activation.Activation;
import net.echo.math4j.math.tensor.Tensor;

public class SoftmaxActivation implements Activation {

    @Override
    public double activate(double input) {
        throw new UnsupportedOperationException(
                "Softmax is a vector-based activation; use activate(double[]).");
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
