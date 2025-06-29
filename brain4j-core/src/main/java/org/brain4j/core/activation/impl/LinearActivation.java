package org.brain4j.core.activation.impl;

import org.brain4j.core.weightsinit.UniformXavierInit;
import org.brain4j.common.activation.Activation;
import org.brain4j.common.weightsinit.WeightInitialization;

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
