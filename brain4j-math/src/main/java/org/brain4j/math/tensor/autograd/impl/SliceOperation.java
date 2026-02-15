package org.brain4j.math.tensor.autograd.impl;

import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;
import org.brain4j.math.tensor.impl.GpuTensor;
import org.brain4j.math.commons.Range;

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
        // TODO: make a GPU-specialized version
        if (gradOutput instanceof GpuTensor) {
            throw Commons.illegalState("Backward for Slice Op is not supported yet for GPU!");
        }

        Tensor input = inputs[0];
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