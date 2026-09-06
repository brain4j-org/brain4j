package org.brain4j.math.clipper.impl;

import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.kernel.KernelFactory;
import org.brain4j.math.tensor.impl.CpuTensor;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.silicon.api.kernel.ComputeSize;

public class HardClipper implements GradientClipper {

    private double bound;
    
    public HardClipper() {
    }
    
    public HardClipper(double bound) { this.bound = bound; }
    
    @Override
    public void clipCpu(CpuTensor grad) {
        grad.map(x -> Math.clamp(x, -bound, bound));
    }

    @Override
    public void clipGpu(GpuTensor grad) {
        Device device = grad.getDevice();

        try (GpuContext.QueueHandle queue = GpuContext.getOrCreateQueue(device)) {
            ComputeSize size = new ComputeSize(grad.size(), 1, 1);
            KernelFactory.create(device, kernelName())
                .buffer(grad.getDataBuffer())
                .floatVal((float) bound)
                .intVal(grad.size())
                .launch(queue.queue(), size);
        }
    }

    @Override
    public String kernelName() {
        return "hard_clip";
    }
    
    public double bound() {
        return bound;
    }
}
