package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weights.WeightInitialization;
import org.brain4j.math.weights.impl.UniformXavierInit;
import org.jocl.cl_program;

import static org.jocl.CL.clCreateKernel;

public class LinearActivation extends Activation {

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
