package org.brain4j.math.clipper.impl;

import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.tensor.impl.CpuTensor;
import org.brain4j.math.tensor.impl.SiliconGpuTensor;

public class NoClipper implements GradientClipper {
    
    @Override
    public void clipCpu(CpuTensor grad) {
    }

    @Override
    public void clipGpu(SiliconGpuTensor grad) {
    }

    @Override
    public String kernelName() {
        return "";
    }
}