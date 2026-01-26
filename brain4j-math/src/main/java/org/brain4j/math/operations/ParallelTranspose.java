package org.brain4j.math.operations;

import org.brain4j.math.tensor.impl.BaseTensor;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

import static org.brain4j.math.Tensors.PARALLELISM;
import static org.brain4j.math.Tensors.SPLIT_COMPLEXITY_THRESHOLD;

public class ParallelTranspose extends RecursiveAction {
    
    public record TransposeParameters(
        float[] srcData,
        float[] dstData,
        int[] dstShape,
        int[] srcStride,
        int[] dstStride,
        int[] destToSrc
    ) { }
    
    private static boolean isOverSplitThreshold(int work) {
        return work > SPLIT_COMPLEXITY_THRESHOLD;
    }
    
    private final TransposeParameters parameters;
    private final int start;
    private final int end;

    public ParallelTranspose(TransposeParameters parameters, int start, int end) {
        this.parameters = parameters;
        this.start = start;
        this.end  = end;
    }
    
    @Override
    protected void compute() {
        int work = end - start;
        if (isOverSplitThreshold(work)) {
            int mid = (start + end) >>> 1;
            invokeAll(
                new ParallelTranspose(parameters, start, mid),
                new ParallelTranspose(parameters, mid, end)
            );
        } else {
            for (int i = start; i < end; i++) {
                int sOffset = i * parameters.srcStride[parameters.destToSrc[0]];
                int dOffset = i * parameters.dstStride[0];
                copyRecursive(1, sOffset, dOffset);
            }
        }
    }
    
    private void copyRecursive(int dim, int srcOffset, int dstOffset) {
        var params = parameters;
        if (dim == params.dstShape.length) {
            params.dstData[dstOffset] = params.srcData[srcOffset];
            return;
        }
        
        int sStride = params.srcStride[params.destToSrc[dim]];
        int dStride = params.dstStride[dim];
        int extent = params.dstShape[dim];
        
        int s = srcOffset;
        int d = dstOffset;
        for (int i = 0; i < extent; i++) {
            copyRecursive(dim + 1, s, d);
            s += sStride;
            d += dStride;
        }
    }
    
    public static void transpose(BaseTensor source, BaseTensor result, int dim1, int dim2) {
        int rank = source.shape().length;

        int[] dstShape = result.shape();
        int[] srcStride = source.strides();
        int[] dstStride = result.strides();

        int[] destToSrc = new int[rank];
        for (int d = 0; d < rank; d++) destToSrc[d] = d;
        destToSrc[dim1] = dim2;
        destToSrc[dim2] = dim1;

        float[] srcData = source.data();
        float[] dstData = result.data();
        int work = dstShape[0];

        int step = work / PARALLELISM;
        var params = new TransposeParameters(srcData, dstData, dstShape, srcStride, dstStride, destToSrc);
        List<ParallelTranspose> actions = new ArrayList<>();

        for (int i = 0; i < PARALLELISM; i++) {
            int startIndex = i * step;
            int endIndex = (i == PARALLELISM - 1) ? work : Math.min(startIndex + step, work);

            if (startIndex < endIndex) {
                actions.add(new ParallelTranspose(params, startIndex, endIndex));
            }
        }

        ForkJoinTask.invokeAll(actions);
    }
}
