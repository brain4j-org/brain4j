package net.echo.brain4j.activation.impl;

import net.echo.brain4j.activation.Activation;

public class ReLUActivation implements Activation {

    @Override
    public double activate(double input) {
        return Math.max(0, input);
    }

    @Override
    public double getDerivative(double input) {
        return input > 0 ? 1 : 0;
    }
}
