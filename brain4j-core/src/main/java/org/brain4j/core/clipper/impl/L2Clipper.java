package org.brain4j.core.clipper.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.impl.cpu.CpuTensor;
import org.brain4j.common.tensor.impl.gpu.GpuTensor;
import org.brain4j.core.clipper.GradientClipper;

public class L2Clipper implements GradientClipper {

    private final double scale;

    public L2Clipper(double scale) { this.scale = scale; }

    @Override
    public void clipCpu(CpuTensor grad) {
        double threshold = scale * Math.sqrt(grad.elements());
        double norm = sumOfSquares(grad);

        if (norm > threshold) {
            double scaleFactor = threshold / norm;
            grad.mul(scaleFactor);
        }
    }

    @Override
    public void clipGpu(GpuTensor grad) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public String kernelName() {
        return "l2_clip";
    }

    public double sumOfSquares(Tensor input) {
        double sumOfSquares = 0.0;

        for (int i = 0; i < input.elements(); i++) {
            sumOfSquares += Math.pow(input.data()[i], 2);
        }

        return Math.sqrt(sumOfSquares);
    }
}