package org.brain4j.math.tensor.autograd.impl;

import org.brain4j.math.Tensors;
import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.kernel.KernelFactory;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.TensorKey;
import org.brain4j.math.tensor.autograd.Operation;
import org.brain4j.math.tensor.Usage;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.brain4j.math.commons.Range;
import org.silicon.api.device.ComputeBuffer;

public record SliceOperation(Range... ranges) implements Operation {

    @Override
    public int requiredInputs() {
        return 1;
    }

    @Override
    public Tensor compute(Tensor... inputs) {
        return inputs[0].slice(ranges);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        Tensor input = inputs[0];

        if (input instanceof GpuTensor gpuInput && gradOutput instanceof GpuTensor gpuGradOutput) {
            return new Tensor[] { backwardGpu(gpuInput, gpuGradOutput) };
        }

        int[] inputShape = input.shape();

        Tensor gradInput = Tensors.zeros(inputShape);
        Range[] usedRanges = new Range[inputShape.length];

        for (int d = 0; d < inputShape.length; d++) {
            usedRanges[d] = (d < ranges.length) ? ranges[d] : null;
        }

        int[] expectedYShape = new int[inputShape.length];

        for (int d = 0; d < inputShape.length; d++) {
            int dim = inputShape[d];
            Range used = usedRanges[d];
            expectedYShape[d] = used != null ? used.size(dim) : dim;
        }

        gradOutput = gradOutput.reshape(expectedYShape);

        int[] srcIndices = new int[inputShape.length];
        int[] dstIndices = new int[inputShape.length];
        sliceBackwardCopy(gradInput, gradOutput, usedRanges, srcIndices, dstIndices, 0);

        return new Tensor[] { gradInput };
    }

    private Tensor backwardGpu(GpuTensor input, GpuTensor gradOutput) {
        int[] inputShape = input.shape();
        Range[] usedRanges = usedRanges(inputShape);
        int[] expectedOutputShape = expectedOutputShape(inputShape, usedRanges);

        GpuTensor reshapedGradOutput = (GpuTensor) gradOutput.reshape(expectedOutputShape);
        Device device = input.getDevice();
        GpuTensor gradInput = new GpuTensor(device, inputShape);
        gradInput.getDataBuffer().write(new float[gradInput.size()]);

        int[] starts = new int[inputShape.length];
        int[] steps = new int[inputShape.length];
        for (int d = 0; d < inputShape.length; d++) {
            Range range = usedRanges[d];
            starts[d] = range == null ? 0 : range.start(inputShape[d]);
            steps[d] = range == null ? 1 : range.step();
        }

        TensorKey outputShapeKey = new TensorKey(Usage.OTHER, expectedOutputShape);
        TensorKey startsKey = new TensorKey(Usage.OTHER, starts);
        TensorKey stepsKey = new TensorKey(Usage.OTHER, steps);

        ComputeBuffer outputShapeBuffer = device.acquire(outputShapeKey, () -> device.createBuffer(expectedOutputShape));
        ComputeBuffer startsBuffer = device.acquire(startsKey, () -> device.createBuffer(starts));
        ComputeBuffer stepsBuffer = device.acquire(stepsKey, () -> device.createBuffer(steps));

        int[] scatterStrides = org.brain4j.math.Tensors.computeStrides(inputShape);
        try (GpuContext.QueueHandle queue = GpuContext.getOrCreateQueue(device)) {
            KernelFactory.create(device, "slice_backward_scatter")
                .buffer(reshapedGradOutput.getDataBuffer())
                .buffer(gradInput.getDataBuffer())
                .buffer(device.createBuffer(scatterStrides))
                .buffer(reshapedGradOutput.getStridesBuffer())
                .buffer(outputShapeBuffer)
                .buffer(startsBuffer)
                .buffer(stepsBuffer)
                .intVal(inputShape.length)
                .launch(queue.queue(), Math.max(1, reshapedGradOutput.elements()));
        }

        return gradInput;
    }

    private Range[] usedRanges(int[] inputShape) {
        Range[] usedRanges = new Range[inputShape.length];
        for (int d = 0; d < inputShape.length; d++) {
            usedRanges[d] = (d < ranges.length) ? ranges[d] : null;
        }
        return usedRanges;
    }

    private int[] expectedOutputShape(int[] inputShape, Range[] usedRanges) {
        int[] expectedShape = new int[inputShape.length];
        for (int d = 0; d < inputShape.length; d++) {
            Range used = usedRanges[d];
            expectedShape[d] = used != null ? used.size(inputShape[d]) : inputShape[d];
        }
        return expectedShape;
    }

    private void sliceBackwardCopy(
        Tensor gradInput,
        Tensor gradOutput,
        Range[] ranges,
        int[] srcIndices,
        int[] dstIndices,
        int dim
    ) {
        int dims = srcIndices.length;
        int[] gradInputShape = gradInput.shape();
        float[] gIn  = gradInput.data();
        float[] gOut = gradOutput.data();

        if (dim == dims) {
            int idxIn  = gradInput.linearIndex(srcIndices);
            int idxOut = gradOutput.linearIndex(dstIndices);
            gIn[idxIn] += gOut[idxOut];
            return;
        }

        Range range = ranges[dim];
        int size = gradInputShape[dim];
        int start = (range == null) ? 0 : range.start(size);
        int end = (range == null) ? size : range.end(size);
        int step = (range == null) ? 1 : range.step();

        if (dim == dims - 1 && step == 1) {
            int blockLen = end - start;

            srcIndices[dim] = start;
            dstIndices[dim] = 0;

            int offIn = gradInput.linearIndex(srcIndices);
            int offOut = gradOutput.linearIndex(dstIndices);

            for (int k = 0; k < blockLen; k++) {
                gIn[offIn + k] += gOut[offOut + k];
            }
            return;
        }

        for (int i = start, j = 0; i < end; i += step, j++) {
            srcIndices[dim] = i;
            dstIndices[dim] = j;
            sliceBackwardCopy(gradInput, gradOutput, ranges, srcIndices, dstIndices, dim + 1);
        }
    }
}
