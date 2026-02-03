package org.brain4j.math.clipper.impl;

import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.gpu.silicon.SiliconContext;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.gpu.silicon.SiliconKernel;
import org.brain4j.math.tensor.impl.CpuTensor;
import org.brain4j.math.tensor.impl.SiliconGpuTensor;
import org.silicon.api.kernel.ComputeSize;

public class HardClipper implements GradientClipper {

    private double bound;
    
    public HardClipper() {
    }
    
    public HardClipper(double bound) { this.bound = bound; }
    
    @Override
    public void clipCpu(CpuTensor grad) {
        grad.map(x -> Commons.clamp(x, -bound, bound));
    }

    @Override
    public void clipGpu(SiliconGpuTensor grad) {
        SiliconDevice device = grad.device();

        try (SiliconContext.QueueHandle queue = SiliconContext.getOrCreateQueue(device)) {
            ComputeSize size = new ComputeSize(grad.size(), 1, 1);
            SiliconKernel.create(device, kernelName())
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
}