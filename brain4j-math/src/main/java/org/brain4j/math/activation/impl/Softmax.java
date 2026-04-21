package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.weightsinit.impl.UniformXavierInit;
import org.brain4j.math.weightsinit.WeightInit;

public record Softmax(double temperature) implements Activation {

    public Softmax() {
        this(1.0);
    }

    @Override
    public WeightInit defaultWeightInit() {
        return new UniformXavierInit();
    }

    @Override
    public double activate(double input) {
        return 1.0;
    }

    @Override
    public Tensor activate(Tensor input) {
        return input.softmax(temperature);
    }

    @Override
    public double derivative(double input) {
        return 0.0;
    }

    @Override
    public String getKernelPrefix() {
        return "";
    }

    @Override
    public Tensor derivative(Tensor input, Tensor output, Tensor gradOut) {
        if (gradOut == null) return null;
        
        Tensor dot = gradOut.mul(output).sum(-1, true);
        return output.mul(gradOut.minus(dot));
    }

    @Override
    public int getActivationId() {
        return -1; // we got a custom kernel for softmax
    }
}
