package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.gpu.silicon.SiliconKernel;
import org.brain4j.math.tensor.impl.SiliconGpuTensor;
import org.brain4j.math.weightsinit.impl.NormalHeInit;
import org.brain4j.math.weightsinit.WeightInit;
import org.silicon.api.function.ComputeFunction;

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
    public SiliconKernel createKernel(ComputeFunction kernel, SiliconGpuTensor current, SiliconGpuTensor other) {
        return SiliconKernel.create(kernel)
            .buffer(current.getDataBuffer())
            .buffer(other.getDataBuffer())
            .floatVal((float) alpha)
            .intVal(current.size());
    }
}
