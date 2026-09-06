package org.brain4j.math.convolution.im2col;

import java.util.concurrent.RecursiveAction;

public class Im2ColTask extends RecursiveAction {

    private static final int PARALLEL_THRESHOLD = 1 << 12; // 4096 patch
    private final Im2ColParams params;
    private final int startPatch;
    private final int endPatch;

    public Im2ColTask(Im2ColParams params, int startPatch, int endPatch) {
        this.params = params;
        this.startPatch = startPatch;
        this.endPatch = endPatch;
    }

    @Override
    protected void compute() {
        int patchSize = params.channelCount() * params.filterHeight() * params.filterWidth();
        int work = endPatch - startPatch;

        if (work > PARALLEL_THRESHOLD) {
            int mid = (startPatch + endPatch) >>> 1;
            invokeAll(
                new Im2ColTask(params, startPatch, mid),
                new Im2ColTask(params, mid, endPatch)
            );
            return;
        }

        float[] inputData = params.inputData();
        float[] resultData = params.resultData();
        int inputBase = params.inputBaseOffset();
        int resultBase = params.resultBaseOffset();

        int filterSize = params.filterHeight() * params.filterWidth();
        int inputSize = params.inputHeight() * params.inputWidth();

        for (int patchIndex = startPatch; patchIndex < endPatch; patchIndex++) {
            int outRow = patchIndex / params.outputWidth();
            int outCol = patchIndex % params.outputWidth();
            int baseResultOffset = resultBase + patchIndex * patchSize;

            for (int c = 0; c < params.channelCount(); c++) {
                int channelOffsetInput = inputBase + c * inputSize + outCol;
                int channelOffsetResult = c * filterSize + baseResultOffset;

                for (int fh = 0; fh < params.filterHeight(); fh++) {
                    int srcPos = (outRow + fh) * params.inputWidth() + channelOffsetInput;
                    int destPos = channelOffsetResult + fh * params.filterWidth();
                    System.arraycopy(inputData, srcPos, resultData, destPos, params.filterWidth());
                }
            }
        }
    }
}