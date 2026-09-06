package org.brain4j.math.gpu.ops;

import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.kernel.KernelFactory;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.silicon.api.kernel.ComputeSize;

import java.util.Arrays;

public final class FlashAttention {

    private static final int TILE_SIZE = 16;

    private FlashAttention() {}

    public record ForwardResult(GpuTensor output, GpuTensor lse) {}

    public record BackwardResult(GpuTensor dQ, GpuTensor dK, GpuTensor dV) {}

    public static GpuTensor forward(
        GpuTensor query,
        GpuTensor key,
        GpuTensor value,
        float scale,
        boolean causal
    ) {
        ShapeInfo shape = validateForwardInputs(query, key, value);
        GpuTensor output = new GpuTensor(query.getDevice(), query.shape());

        dispatchForward("flash_attention_forward", query, key, value, output, null, shape, scale, causal);
        return output;
    }

    public static ForwardResult forwardWithLse(
        GpuTensor query,
        GpuTensor key,
        GpuTensor value,
        float scale,
        boolean causal
    ) {
        ShapeInfo shape = validateForwardInputs(query, key, value);
        GpuTensor output = new GpuTensor(query.getDevice(), query.shape());
        GpuTensor lse = new GpuTensor(query.getDevice(), new int[] { shape.batchSize, shape.numHeads, shape.seqLen });

        dispatchForward("flash_attention_forward_with_lse", query, key, value, output, lse, shape, scale, causal);
        return new ForwardResult(output, lse);
    }

    public static ForwardResult forwardTiled(
        GpuTensor query,
        GpuTensor key,
        GpuTensor value,
        float scale,
        boolean causal
    ) {
        ShapeInfo shape = validateForwardInputs(query, key, value);
        if (shape.headDim > 64) {
            throw new IllegalArgumentException("Tiled FlashAttention supports headDim <= 64, got " + shape.headDim);
        }

        GpuTensor output = new GpuTensor(query.getDevice(), query.shape());
        GpuTensor lse = new GpuTensor(query.getDevice(), new int[] { shape.batchSize, shape.numHeads, shape.seqLen });

        dispatchForward("flash_attention_forward_tiled", query, key, value, output, lse, shape, scale, causal);
        return new ForwardResult(output, lse);
    }

    public static BackwardResult backward(
        GpuTensor query,
        GpuTensor key,
        GpuTensor value,
        GpuTensor output,
        GpuTensor gradOutput,
        GpuTensor lse,
        float scale,
        boolean causal
    ) {
        ShapeInfo shape = validateBackwardInputs(query, key, value, output, gradOutput, lse);
        Device device = query.getDevice();
        GpuTensor dQ = new GpuTensor(device, query.shape());
        GpuTensor dK = new GpuTensor(device, key.shape());
        GpuTensor dV = new GpuTensor(device, value.shape());

        dispatchBackwardKV("flash_attention_backward", query, key, value, output, gradOutput, lse, dK, dV, shape, scale, causal);
        dispatchBackwardQ("flash_attention_backward_dq", query, key, value, output, gradOutput, lse, dQ, shape, scale, causal);

        return new BackwardResult(dQ, dK, dV);
    }

    public static BackwardResult backwardTiled(
        GpuTensor query,
        GpuTensor key,
        GpuTensor value,
        GpuTensor output,
        GpuTensor gradOutput,
        GpuTensor lse,
        float scale,
        boolean causal
    ) {
        ShapeInfo shape = validateBackwardInputs(query, key, value, output, gradOutput, lse);
        Device device = query.getDevice();
        GpuTensor dQ = new GpuTensor(device, query.shape());
        GpuTensor dK = new GpuTensor(device, key.shape());
        GpuTensor dV = new GpuTensor(device, value.shape());

        dK.getDataBuffer().write(new float[dK.size()]);
        dV.getDataBuffer().write(new float[dV.size()]);

        try (GpuContext.QueueHandle queue = GpuContext.getOrCreateQueue(device)) {
            KernelFactory.create(device, "flash_attention_backward_tiled")
                .buffer(query.getDataBuffer())
                .buffer(key.getDataBuffer())
                .buffer(value.getDataBuffer())
                .buffer(output.getDataBuffer())
                .buffer(gradOutput.getDataBuffer())
                .buffer(lse.getDataBuffer())
                .buffer(dQ.getDataBuffer())
                .buffer(dK.getDataBuffer())
                .buffer(dV.getDataBuffer())
                .buffer(query.getStridesBuffer())
                .buffer(key.getStridesBuffer())
                .buffer(value.getStridesBuffer())
                .buffer(output.getStridesBuffer())
                .buffer(gradOutput.getStridesBuffer())
                .buffer(dQ.getStridesBuffer())
                .buffer(dK.getStridesBuffer())
                .buffer(dV.getStridesBuffer())
                .intVal(shape.batchSize)
                .intVal(shape.numHeads)
                .intVal(shape.seqLen)
                .intVal(shape.headDim)
                .floatVal(scale)
                .intVal(causal ? 1 : 0)
                .launch(queue.queue(), global(shape));
        }

        return new BackwardResult(dQ, dK, dV);
    }

    private static void dispatchForward(
        String kernelName,
        GpuTensor query,
        GpuTensor key,
        GpuTensor value,
        GpuTensor output,
        GpuTensor lse,
        ShapeInfo shape,
        float scale,
        boolean causal
    ) {
        Device device = query.getDevice();
        try (GpuContext.QueueHandle queue = GpuContext.getOrCreateQueue(device)) {
            KernelFactory kernel = KernelFactory.create(device, kernelName)
                .buffer(query.getDataBuffer())
                .buffer(key.getDataBuffer())
                .buffer(value.getDataBuffer())
                .buffer(output.getDataBuffer());

            if (lse != null) {
                kernel.buffer(lse.getDataBuffer());
            }

            kernel.buffer(query.getStridesBuffer())
                .buffer(key.getStridesBuffer())
                .buffer(value.getStridesBuffer())
                .buffer(output.getStridesBuffer())
                .intVal(shape.batchSize)
                .intVal(shape.numHeads)
                .intVal(shape.seqLen)
                .intVal(shape.headDim)
                .floatVal(scale)
                .intVal(causal ? 1 : 0);

            if ("flash_attention_forward_tiled".equals(kernelName)) {
                kernel.launch(queue.queue(), roundUpGlobal(shape), new ComputeSize(TILE_SIZE, 1, 1));
            } else {
                kernel.launch(queue.queue(), global(shape));
            }
        }
    }

    private static void dispatchBackwardKV(
        String kernelName,
        GpuTensor query,
        GpuTensor key,
        GpuTensor value,
        GpuTensor output,
        GpuTensor gradOutput,
        GpuTensor lse,
        GpuTensor dK,
        GpuTensor dV,
        ShapeInfo shape,
        float scale,
        boolean causal
    ) {
        Device device = query.getDevice();
        try (GpuContext.QueueHandle queue = GpuContext.getOrCreateQueue(device)) {
            KernelFactory.create(device, kernelName)
                .buffer(query.getDataBuffer())
                .buffer(key.getDataBuffer())
                .buffer(value.getDataBuffer())
                .buffer(output.getDataBuffer())
                .buffer(gradOutput.getDataBuffer())
                .buffer(lse.getDataBuffer())
                .buffer(dK.getDataBuffer())
                .buffer(dV.getDataBuffer())
                .buffer(query.getStridesBuffer())
                .buffer(key.getStridesBuffer())
                .buffer(value.getStridesBuffer())
                .buffer(output.getStridesBuffer())
                .buffer(gradOutput.getStridesBuffer())
                .buffer(dK.getStridesBuffer())
                .buffer(dV.getStridesBuffer())
                .intVal(shape.batchSize)
                .intVal(shape.numHeads)
                .intVal(shape.seqLen)
                .intVal(shape.headDim)
                .floatVal(scale)
                .intVal(causal ? 1 : 0)
                .launch(queue.queue(), global(shape));
        }
    }

    private static void dispatchBackwardQ(
        String kernelName,
        GpuTensor query,
        GpuTensor key,
        GpuTensor value,
        GpuTensor output,
        GpuTensor gradOutput,
        GpuTensor lse,
        GpuTensor dQ,
        ShapeInfo shape,
        float scale,
        boolean causal
    ) {
        Device device = query.getDevice();
        try (GpuContext.QueueHandle queue = GpuContext.getOrCreateQueue(device)) {
            KernelFactory.create(device, kernelName)
                .buffer(query.getDataBuffer())
                .buffer(key.getDataBuffer())
                .buffer(value.getDataBuffer())
                .buffer(output.getDataBuffer())
                .buffer(gradOutput.getDataBuffer())
                .buffer(lse.getDataBuffer())
                .buffer(dQ.getDataBuffer())
                .buffer(query.getStridesBuffer())
                .buffer(key.getStridesBuffer())
                .buffer(value.getStridesBuffer())
                .buffer(output.getStridesBuffer())
                .buffer(gradOutput.getStridesBuffer())
                .buffer(dQ.getStridesBuffer())
                .intVal(shape.batchSize)
                .intVal(shape.numHeads)
                .intVal(shape.seqLen)
                .intVal(shape.headDim)
                .floatVal(scale)
                .intVal(causal ? 1 : 0)
                .launch(queue.queue(), global(shape));
        }
    }

    private static ShapeInfo validateForwardInputs(
        GpuTensor query,
        GpuTensor key,
        GpuTensor value
    ) {
        requireSameDevice(query, key, value);
        requireRank4("query", query);
        requireRank4("key", key);
        requireRank4("value", value);

        int[] qShape = query.shape();
        int[] kShape = key.shape();
        int[] vShape = value.shape();

        if (qShape[0] != kShape[0] || qShape[0] != vShape[0]
            || qShape[1] != kShape[1] || qShape[1] != vShape[1]
            || kShape[2] != vShape[2]
            || qShape[3] != kShape[3] || qShape[3] != vShape[3]) {
            throw new IllegalArgumentException(
                "FlashAttention expects Q, K, V shapes [batch, heads, seq, headDim] with matching batch/head/headDim. " +
                "Got Q=" + Arrays.toString(qShape) + ", K=" + Arrays.toString(kShape) + ", V=" + Arrays.toString(vShape)
            );
        }

        if (qShape[2] != kShape[2]) {
            throw new IllegalArgumentException("Only equal query/key sequence length is currently supported");
        }

        return new ShapeInfo(qShape[0], qShape[1], qShape[2], qShape[3]);
    }

    private static ShapeInfo validateBackwardInputs(
        GpuTensor query,
        GpuTensor key,
        GpuTensor value,
        GpuTensor output,
        GpuTensor gradOutput,
        GpuTensor lse
    ) {
        ShapeInfo shape = validateForwardInputs(query, key, value);
        requireSameDevice(query, output, gradOutput, lse);
        requireShape("output", output, query.shape());
        requireShape("gradOutput", gradOutput, query.shape());
        requireShape("lse", lse, new int[] { shape.batchSize, shape.numHeads, shape.seqLen });
        return shape;
    }

    private static void requireRank4(String name, GpuTensor tensor) {
        if (tensor.rank() != 4) {
            throw new IllegalArgumentException(name + " must have shape [batch, heads, seq, headDim]");
        }
    }

    private static void requireShape(String name, GpuTensor tensor, int[] expectedShape) {
        if (!Arrays.equals(tensor.shape(), expectedShape)) {
            throw new IllegalArgumentException(
                name + " shape mismatch. Expected " + Arrays.toString(expectedShape) +
                ", got " + Arrays.toString(tensor.shape())
            );
        }
    }

    private static void requireSameDevice(GpuTensor first, GpuTensor... others) {
        for (GpuTensor other : others) {
            if (!first.getDevice().equals(other.getDevice())) {
                throw new IllegalArgumentException("All FlashAttention tensors must be on the same GPU device");
            }
        }
    }

    private static ComputeSize global(ShapeInfo shape) {
        return new ComputeSize(shape.seqLen, shape.numHeads, shape.batchSize);
    }

    private static ComputeSize roundUpGlobal(ShapeInfo shape) {
        int roundedSeq = ((shape.seqLen + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE;
        return new ComputeSize(roundedSeq, shape.numHeads, shape.batchSize);
    }

    private record ShapeInfo(int batchSize, int numHeads, int seqLen, int headDim) {}
}
