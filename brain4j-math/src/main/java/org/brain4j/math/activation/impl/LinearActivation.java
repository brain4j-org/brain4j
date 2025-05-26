package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weights.WeightInitialization;
import org.brain4j.math.weights.impl.NormalInit;

public class LinearActivation implements Activation {

    @Override
    public WeightInitialization defaultWeightInit() {
        return new NormalInit();
    }

    @Override
    public double activate(double input) {
        return input;
    }

    @Override
    public double getDerivative(double input) {
        return 1;
    }
}
