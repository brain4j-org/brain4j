package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weights.WeightInitialization;
import org.brain4j.math.weights.impl.NormalHeInit;

public class ReLUActivation implements Activation {

    @Override
    public WeightInitialization defaultWeightInit() {
        return new NormalHeInit();
    }

    @Override
    public double activate(double input) {
        return Math.max(0, input);
    }

    @Override
    public double getDerivative(double input) {
        return input > 0 ? 1 : 0;
    }
}
