package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weights.WeightInitialization;
import org.brain4j.math.weights.impl.NormalXavierInit;
import org.jocl.cl_program;

public class TanhActivation extends Activation {

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
