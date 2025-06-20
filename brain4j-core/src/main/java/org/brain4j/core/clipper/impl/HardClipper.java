package org.brain4j.core.clipper.impl;

import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.math.kernel.GpuKernelCache;
import org.brain4j.math.kernel.KernelFactory;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.cpu.CpuTensor;
import org.brain4j.math.tensor.impl.gpu.GpuTensor;
import org.brain4j.math.tensor.impl.gpu.OpenCLContext;
import org.jocl.cl_command_queue;
import org.jocl.cl_kernel;

public class HardClipper implements GradientClipper {

    private final double bound;

    public HardClipper(double bound) { this.bound = bound; }

    @Override
    public void clipCpu(CpuTensor grad) {
        float[] gradData = grad.data();

        for (int i = 0; i < grad.elements(); i++) {
            double clamped = Math.max(-bound, Math.min(bound, gradData[i]));
            gradData[i] = (float) clamped;
        }
    }

    @Override
    public void clipGpu(GpuTensor grad) {
        cl_kernel kernel = GpuKernelCache.kernel(kernelName());
        cl_command_queue queue = OpenCLContext.currentQueue();

        KernelFactory
            .create(kernel)
            .addMemParam(grad.dataBuffer())
            .addFloatParam((float) bound)
            .addIntParam(grad.size())
            .launch(queue, 1, grad.size());
    }

    @Override
    public String kernelName() {
        return "hard_clip";
    }
}