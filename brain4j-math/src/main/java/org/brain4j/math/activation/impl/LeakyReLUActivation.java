package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weights.WeightInitialization;
import org.brain4j.math.weights.impl.NormalHeInit;
import org.jocl.cl_program;

import static org.jocl.CL.clCreateKernel;

public class LeakyReLUActivation extends Activation {

    private final double alpha;

    public LeakyReLUActivation() {
        this(0.01);
    }

    public LeakyReLUActivation(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public WeightInitialization defaultWeightInit() {
        return new NormalHeInit();
    }

    @Override
    public double activate(double input) {
        return Math.max(alpha * input, input);
    }

    @Override
    public double derivative(double input) {
        return input > 0 ? 1 : alpha;
    }

    @Override
    public String kernelPrefix() {
        return "leakyrelu";
    }
}
