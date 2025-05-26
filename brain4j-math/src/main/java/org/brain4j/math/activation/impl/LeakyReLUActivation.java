package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weights.WeightInitialization;
import org.brain4j.math.weights.impl.NormalHeInit;

public class LeakyReLUActivation implements Activation {

    @Override
    public WeightInitialization defaultWeightInit() {
        return new NormalHeInit();
    }

    @Override
    public double activate(double input) {
        return Math.max(0.01 * input, input);
    }

    @Override
    public double getDerivative(double input) {
        return input > 0 ? 1 : 0.01;
    }
}
