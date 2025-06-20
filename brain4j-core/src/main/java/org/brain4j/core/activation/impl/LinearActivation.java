package org.brain4j.core.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weightsinit.WeightInitialization;
import org.brain4j.core.weightsinit.UniformXavierInit;

public class LinearActivation implements Activation {

    @Override
    public WeightInitialization defaultWeightInit() {
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
    public String kernelPrefix() {
        return "linear";
    }
}
