package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.gpu.kernel.KernelFactory;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.brain4j.math.weightsinit.impl.NormalHeInit;
import org.brain4j.math.weightsinit.WeightInit;

public record ELUActivation(double alpha) implements Activation {

    public ELUActivation() {
        this(1.0);
    }

    @Override
    public WeightInit defaultWeightInit() {
        return new NormalHeInit();
    }

    @Override
    public double activate(double input) {
        return input > 0 ? input : (alpha * Math.exp(input) - 1);
    }

    @Override
    public double derivative(double input) {
        return input > 0 ? 1 : (alpha * Math.exp(input));
    }

    @Override
    public String getKernelPrefix() {
        return "elu";
    }

    @Override
    public KernelFactory createKernel(long kernel, GpuTensor current, GpuTensor other) {
        return KernelFactory.create(kernel)
            .addMemParam(current.dataBuffer())
            .addMemParam(other.dataBuffer())
            .addFloatParam((float) alpha)
            .addIntParam(current.size());
    }
}
