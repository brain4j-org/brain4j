package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weightsinit.impl.UniformXavierInit;
import org.brain4j.math.weightsinit.WeightInit;

public class SILUActivation implements Activation {

    @Override
    public WeightInit defaultWeightInit() {
        return new UniformXavierInit();
    }

    @Override
    public double activate(double input) {
        double sig = 1.0 / (1.0 + Math.exp(-input));
        return input * sig;
    }

    @Override
    public double derivative(double input) {
        double sig = 1.0 / (1.0 + Math.exp(-input));
        return sig * (1.0 + input * (1.0 - sig));
    }

    @Override
    public String getKernelPrefix() {
        return "silu"; // TODO
    }
}
