package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.gpu.kernel.KernelFactory;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.brain4j.math.weightsinit.impl.NormalHeInit;
import org.brain4j.math.weightsinit.WeightInit;

public record LeakyReLUActivation(double alpha) implements Activation {

    public LeakyReLUActivation() {
        this(0.01);
    }

    @Override
    public WeightInit defaultWeightInit() {
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
    public String getKernelPrefix() {
        return "leakyrelu";
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
