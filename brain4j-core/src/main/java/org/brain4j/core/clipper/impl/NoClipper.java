package org.brain4j.core.clipper.impl;

import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.common.tensor.impl.cpu.CpuTensor;
import org.brain4j.common.tensor.impl.gpu.GpuTensor;

public class NoClipper implements GradientClipper {

    @Override
    public void clipCpu(CpuTensor grad) {

    }

    @Override
    public void clipGpu(GpuTensor grad) {

    }

    @Override
    public String kernelName() {
        return "";
    }
}