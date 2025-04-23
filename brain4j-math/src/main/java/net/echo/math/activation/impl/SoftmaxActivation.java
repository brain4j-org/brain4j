package net.echo.math.activation.impl;

import net.echo.math.activation.Activation;
import net.echo.math.tensor.Tensor;

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
