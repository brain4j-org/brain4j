package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.gpu.silicon.SiliconKernel;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.SiliconGpuTensor;
import org.brain4j.math.weightsinit.impl.UniformXavierInit;
import org.brain4j.math.weightsinit.WeightInit;
import org.silicon.api.function.ComputeFunction;

public record SoftmaxActivation(double temperature) implements Activation {

    public SoftmaxActivation() {
        this(1.0);
    }

    @Override
    public WeightInit defaultWeightInit() {
        return new UniformXavierInit();
    }

    @Override
    public double activate(double input) {
        throw new UnsupportedOperationException("Softmax is a vector-based activation");
    }

    @Override
    public Tensor activate(Tensor input) {
        return input.softmax(temperature);
    }

    @Override
    public double derivative(double input) {
        throw new UnsupportedOperationException("Softmax is a vector-based activation");
    }

    @Override
    public String getKernelPrefix() {
        return "";
    }

    @Override
    public Tensor derivative(Tensor input, Tensor output, Tensor gradOut) {
        Tensor dot = gradOut.mul(output).sum(-1, true);
        return output.mul(gradOut.minus(dot));
    }

    @Override
    public int getActivationId() {
        return -1; // we got a custom kernel for softmax
    }
}
