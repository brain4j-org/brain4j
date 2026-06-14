package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.gpu.kernel.KernelFactory;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.brain4j.math.weightsinit.impl.NormalHeInit;
import org.brain4j.math.weightsinit.WeightInit;
import org.silicon.api.function.ComputeFunction;

public record LeakyReLU(double alpha) implements Activation {

    public LeakyReLU() {
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
    public int getActivationId() {
        return 1;
    }

    @Override
    public KernelFactory createKernel(ComputeFunction kernel, GpuTensor input, GpuTensor output) {
        return KernelFactory.create(kernel)
            .intVal(getActivationId())
            .floatVal((float) alpha)
            .intVal(input.size())
            .buffer(input.getDataBuffer())
            .buffer(output.getDataBuffer());
    }
}
