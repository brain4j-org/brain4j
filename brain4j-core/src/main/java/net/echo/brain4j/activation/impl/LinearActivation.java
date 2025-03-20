package net.echo.brain4j.activation.impl;

import net.echo.brain4j.activation.Activation;

public class LinearActivation implements Activation {

    @Override
    public double activate(double input) {
        return input;
    }

    @Override
    public double getDerivative(double input) {
        return 1;
    }
}
