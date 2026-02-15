package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weightsinit.impl.NormalHeInit;
import org.brain4j.math.weightsinit.WeightInit;

public class ReLUActivation implements Activation {

    @Override
    public WeightInit defaultWeightInit() {
        return new NormalHeInit();
    }

    @Override
    public double activate(double input) {
        return Math.max(0, input);
    }

    @Override
    public double derivative(double input) {
        return input > 0 ? 1 : 0;
    }

    @Override
    public String getKernelPrefix() {
        return "relu";
    }

    @Override
    public int getActivationId() {
        return 0;
    }
}
