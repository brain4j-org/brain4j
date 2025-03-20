package net.echo.brain4j.activation.impl;

import net.echo.brain4j.activation.Activation;

public class SwishActivation implements Activation {

    @Override
    public double activate(double input) {
        return input * (1.0 / (1.0 + Math.exp(-input)));
    }

    @Override
    public double getDerivative(double input) {
        double sigmoid = 1.0 / (1.0 + Math.exp(-input));
        return sigmoid + input * sigmoid * (1 - sigmoid);
    }
}
