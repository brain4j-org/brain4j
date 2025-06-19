package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weights.WeightInitialization;
import org.brain4j.math.weights.impl.NormalHeInit;

public class ELUActivation extends Activation {

    private final double alpha;

    public ELUActivation() {
        this(1.0);
    }

    public ELUActivation(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public WeightInitialization defaultWeightInit() {
        return new NormalHeInit();
    }

    @Override
    public double activate(double input) {
        if (input > 0) {
            return input;
        } else {
            return alpha * (Math.exp(input) - 1);
        }
    }

    @Override
    public double derivative(double input) {
        if (input > 0) {
            return 1;
        } else {
            return alpha * Math.exp(input);
        }
    }

    @Override
    public String kernelPrefix() {
        return "elu";
    }
}
