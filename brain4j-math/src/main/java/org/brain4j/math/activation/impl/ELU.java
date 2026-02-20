package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.gpu.silicon.SiliconKernel;
import org.brain4j.math.tensor.impl.SiliconGpuTensor;
import org.brain4j.math.weightsinit.impl.NormalHeInit;
import org.brain4j.math.weightsinit.WeightInit;
import org.silicon.api.function.ComputeFunction;

public record ELU(double alpha) implements Activation {

    public ELU() {
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
    public int getActivationId() {
        return 2;
    }

    @Override
    public SiliconKernel createKernel(ComputeFunction kernel, SiliconGpuTensor input, SiliconGpuTensor output) {
        return SiliconKernel
            .create(kernel)
            .intVal(getActivationId()) // activation type
            .floatVal((float) alpha) // alpha
            .intVal(input.size()) // length
            .buffer(input.getDataBuffer())
            .buffer(output.getDataBuffer());
    }
}
