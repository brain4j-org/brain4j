package org.brain4j.core.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.kernel.KernelFactory;
import org.brain4j.math.tensor.impl.gpu.GpuTensor;
import org.brain4j.math.weightsinit.WeightInitialization;
import org.brain4j.core.weightsinit.NormalHeInit;
import org.jocl.cl_kernel;

public class ELUActivation implements Activation {

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
        return input > 0 ? input : (alpha * Math.exp(input) - 1);
    }

    @Override
    public double derivative(double input) {
        return input > 0 ? 1 : (alpha * Math.exp(input));
    }

    @Override
    public String kernelPrefix() {
        return "elu";
    }

    @Override
    public KernelFactory createKernel(cl_kernel kernel, GpuTensor current, GpuTensor other) {
        return KernelFactory.create(kernel)
            .addMemParam(current.dataBuffer())
            .addMemParam(other.dataBuffer())
            .addFloatParam((float) alpha)
            .addIntParam(current.size());
    }
}
