package org.brain4j.math.operations;

import org.brain4j.math.Tensors;
import org.brain4j.math.gpu.device.DeviceUtils;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.convolution.ConvolveProvider;
import org.brain4j.math.convolution.impl.NormalConvolveProvider;
import org.brain4j.math.convolution.impl.SIMDConvolveProvider;
import org.brain4j.math.commons.Range;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ForkJoinPool;

public class ParallelConvolve {

    private static final int CORES = Runtime.getRuntime().availableProcessors();
    private static final ForkJoinPool POOL = new ForkJoinPool(CORES);
    
    public static Tensor convolve(Tensor a, Tensor b) {
        while (a.rank() < 4) a = a.unsqueeze();
        while (b.rank() < 4) b = b.unsqueeze();

        ConvolveProvider provider = DeviceUtils.isSimdAvailable()
            ? new SIMDConvolveProvider()
            : new NormalConvolveProvider();

        int[] aShape = a.shape();
        int[] bShape = b.shape();

        boolean aHasBatch = aShape.length == 4;
        int batch = aHasBatch ? aShape[0] : 1;
        int inChannels = aShape[aShape.length - 3];
        int inHeight = aShape[aShape.length - 2];
        int inWidth = aShape[aShape.length - 1];

        int numFilters = bShape[0];
        int filterHeight = bShape[2];
        int filterWidth = bShape[3];

        int outHeight = inHeight - filterHeight + 1;
        int outWidth = inWidth - filterWidth + 1;

        int patchSize = inChannels * filterHeight * filterWidth;
        int totalPatches = outHeight * outWidth;

        Tensor out = Tensors.zeros(batch, numFilters, outHeight, outWidth);
        Tensor filterFlat = b.reshape(numFilters, patchSize);

        float[] filterData = filterFlat.data();
        float[] outData = out.data();

        for (int bIdx = 0; bIdx < batch; bIdx++) {
            Tensor inputBatch = (aHasBatch ? a.slice(Range.point(bIdx)) : a).squeeze(0);
            Tensor patchMatrix = Tensors.im2col(inputBatch, filterHeight, filterWidth);
            float[] patchData = patchMatrix.data(); // [patch_size, total_patches]
            
            List<Callable<Void>> tasks = new ArrayList<>();
            
            for (int f = 0; f < numFilters; f++) {
                int filterOffset = f * patchSize;
                int outBase = (bIdx * numFilters + f) * totalPatches;
                
                int blockSize = 4096;
                int blocks = (totalPatches + blockSize - 1) / blockSize;
                
                ConvolveProvider.PatchData data = new ConvolveProvider.PatchData(
                    filterData, patchData, outData, totalPatches, patchSize, filterOffset, outBase
                );
                
                for (int block = 0; block < blocks; block++) {
                    int finalBlock = block;
                    tasks.add(() -> {
                        int start = finalBlock * blockSize;
                        int end = Math.min(start + blockSize, totalPatches);
                        provider.dotBlock(start, end, data);
                        return null;
                    });
                }
            }
            
            POOL.invokeAll(tasks);
        }

        return out;
    }
}