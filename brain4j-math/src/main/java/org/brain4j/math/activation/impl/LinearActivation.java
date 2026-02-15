package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weightsinit.impl.UniformXavierInit;
import org.brain4j.math.weightsinit.WeightInit;

public class LinearActivation implements Activation {

    @Override
    public WeightInit defaultWeightInit() {
        return new UniformXavierInit();
    }

    @Override
    public double activate(double input) {
        return input;
    }

    @Override
    public double derivative(double input) {
        return 1;
    }

    @Override
    public String getKernelPrefix() {
        return "linear";
    }

    @Override
    public int getActivationId() {
        return 9;
    }
}
