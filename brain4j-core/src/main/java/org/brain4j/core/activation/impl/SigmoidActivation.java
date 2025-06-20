package org.brain4j.core.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weightsinit.WeightInitialization;
import org.brain4j.core.weightsinit.UniformXavierInit;

public class SigmoidActivation implements Activation {

    @Override
    public WeightInitialization defaultWeightInit() {
        return new UniformXavierInit();
    }

    @Override
    public double activate(double input) {
        return 1 / (1 + Math.exp(-input));
    }

    @Override
    public double derivative(double input) {
        double activated = activate(input);
        return activated * (1 - activated);
    }

    @Override
    public String kernelPrefix() {
        return "sigmoid";
    }
}
