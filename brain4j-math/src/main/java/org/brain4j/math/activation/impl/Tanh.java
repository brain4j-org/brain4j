package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weightsinit.impl.UniformXavierInit;
import org.brain4j.math.weightsinit.WeightInit;

public class Tanh implements Activation {

    @Override
    public WeightInit defaultWeightInit() {
        return new UniformXavierInit();
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
    public String getKernelPrefix() {
        return "tanh";
    }

    @Override
    public int getActivationId() {
        return 4;
    }
}
