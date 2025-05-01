package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;

public class ELUActivation implements Activation {

    private final double alpha = 1.0;

    @Override
    public double activate(double input) {
        if (input > 0) {
            return input;
        } else {
            return alpha * (Math.exp(input) - 1);
        }
    }

    @Override
    public double getDerivative(double input) {
        if (input > 0) {
            return 1;
        } else {
            return alpha * Math.exp(input);
        }
    }
}
