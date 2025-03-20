package net.echo.brain4j.activation.impl;

import net.echo.brain4j.activation.Activation;

public class MishActivation implements Activation {

    @Override
    public double activate(double input) {
        double softplus = Math.log1p(Math.exp(input));
        return input * Math.tanh(softplus);
    }

    @Override
    public double getDerivative(double input) {
        double softplus = Math.log1p(Math.exp(input));
        double tanhSp = Math.tanh(softplus);
        double sigmoid = 1.0 / (1.0 + Math.exp(-input));

        return tanhSp + input * sigmoid * (1 - tanhSp * tanhSp);
    }
}
