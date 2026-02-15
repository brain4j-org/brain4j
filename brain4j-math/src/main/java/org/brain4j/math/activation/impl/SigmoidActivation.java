package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weightsinit.impl.UniformXavierInit;
import org.brain4j.math.weightsinit.WeightInit;

public class SigmoidActivation implements Activation {

    @Override
    public WeightInit defaultWeightInit() {
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
    public String getKernelPrefix() {
        return "sigmoid";
    }

    @Override
    public int getActivationId() {
        return 3;
    }
}
