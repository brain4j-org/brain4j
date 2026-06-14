package org.brain4j.math.clipper.impl;

import org.brain4j.math.clipper.GradientClipper;
import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.kernel.KernelFactory;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.CpuTensor;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.silicon.api.device.ComputeBuffer;
import org.silicon.api.kernel.ComputeSize;

public class L2Clipper implements GradientClipper {

    private static final int REDUCTION_BLOCK_SIZE = 256;

    private final double scale;

    public L2Clipper(double scale) { this.scale = scale; }
    
    @Override
    public void clipCpu(CpuTensor grad) {
        double threshold = scale * Math.sqrt(grad.elements());
        double norm = sumOfSquares(grad);

        if (norm > threshold) {
            float scaleFactor = (float) (threshold / norm);
            grad.mul(scaleFactor);
        }
    }

    @Override
    public void clipGpu(GpuTensor grad) {
        int length = grad.size();
        if (length == 0) {
            return;
        }

        Device device = grad.getDevice();
        ComputeBuffer normBuffer = null;
        ComputeBuffer scaleBuffer = null;

        try (GpuContext.QueueHandle queue = GpuContext.getOrCreateQueue(device)) {
            normBuffer = computeNormSquared(queue, device, grad);
            scaleBuffer = device.createBuffer(new float[] { 1.0f });

            float maxNorm = (float) (scale * Math.sqrt(length));

            KernelFactory.create(device, "compute_l2_clip_scale")
                .buffer(normBuffer)
                .buffer(scaleBuffer)
                .floatVal(maxNorm)
                .floatVal(0.0f)
                .launch(queue.queue(), new ComputeSize(1, 1, 1));

            KernelFactory.create(device, "l2_clip_from_scale")
                .buffer(grad.getDataBuffer())
                .buffer(scaleBuffer)
                .intVal(length)
                .launch(queue.queue(), new ComputeSize(length, 1, 1));

            queue.queue().await();
        } finally {
            freeQuietly(scaleBuffer);
            freeQuietly(normBuffer);
        }
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
    
    public double scale() {
        return scale;
    }

    private ComputeBuffer computeNormSquared(
        GpuContext.QueueHandle queue,
        Device device,
        GpuTensor grad
    ) {
        int length = grad.size();
        ComputeSize localSize = new ComputeSize(REDUCTION_BLOCK_SIZE, 1, 1);

        if (length <= REDUCTION_BLOCK_SIZE) {
            ComputeBuffer result = device.createBuffer(new float[] { 0.0f });

            KernelFactory.create(device, "l2_norm_squared_single_block")
                .buffer(grad.getDataBuffer())
                .buffer(result)
                .intVal(length)
                .launch(queue.queue(), localSize, localSize);

            return result;
        }

        int partialCount = ceilDiv(length, REDUCTION_BLOCK_SIZE);
        ComputeBuffer current = device.createBuffer(new float[partialCount]);

        KernelFactory.create(device, "l2_norm_squared_reduce")
            .buffer(grad.getDataBuffer())
            .buffer(current)
            .intVal(length)
            .launch(queue.queue(), new ComputeSize(partialCount * REDUCTION_BLOCK_SIZE, 1, 1), localSize);

        int currentCount = partialCount;
        while (currentCount > REDUCTION_BLOCK_SIZE) {
            int nextCount = ceilDiv(currentCount, REDUCTION_BLOCK_SIZE);
            ComputeBuffer next = device.createBuffer(new float[nextCount]);

            KernelFactory.create(device, "l2_sum_reduce")
                .buffer(current)
                .buffer(next)
                .intVal(currentCount)
                .launch(queue.queue(), new ComputeSize(nextCount * REDUCTION_BLOCK_SIZE, 1, 1), localSize);

            queue.queue().await();
            freeQuietly(current);
            current = next;
            currentCount = nextCount;
        }

        KernelFactory.create(device, "l2_norm_squared_final")
            .buffer(current)
            .intVal(currentCount)
            .launch(queue.queue(), localSize, localSize);

        return current;
    }

    private static int ceilDiv(int value, int divisor) {
        return (value + divisor - 1) / divisor;
    }

    private static void freeQuietly(ComputeBuffer buffer) {
        if (buffer != null && buffer.isAlive()) {
            buffer.free();
        }
    }
}
