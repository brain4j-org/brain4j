package net.echo.math.activation.impl;

import net.echo.math.activation.Activation;

public class SigmoidActivation implements Activation {

    @Override
    public double activate(double input) {
        return 1 / (1 + Math.exp(-input));
    }

    @Override
    public double getDerivative(double input) {
        return input * (1 - input);
    }
}
