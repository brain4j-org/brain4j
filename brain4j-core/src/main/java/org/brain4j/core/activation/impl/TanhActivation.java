package org.brain4j.core.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weightsinit.WeightInitialization;
import org.brain4j.core.weightsinit.NormalXavierInit;

public class TanhActivation implements Activation {

    @Override
    public WeightInitialization defaultWeightInit() {
        return new NormalXavierInit();
    }

    @Override
    public double activate(double input) {
        return Math.tanh(input);
    }

    @Override
    public double derivative(double input) {
        double activated = Math.tanh(input);
        return 1.0 - activated * activated;
    }

    @Override
    public String kernelPrefix() {
        return "tanh";
    }
}
