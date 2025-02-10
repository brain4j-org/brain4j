package net.echo.brain4j.activation.impl;

import net.echo.brain4j.activation.Activation;

public class LeakyReLUActivation implements Activation {

    @Override
    public double activate(double input) {
        return Math.max(0.01 * input, input);
    }

    @Override
    public double getDerivative(double input) {
        return input > 0 ? 1 : 0.01;
    }
}
