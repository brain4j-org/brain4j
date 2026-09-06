package org.brain4j.math.convolution;

public interface ConvolveProvider {
    record PatchData(
        float[] filter,
        float[] patch,
        float[] out,
        int totalPatches,
        int patchSize,
        int filterOffset,
        int outOffset
    ) {}
    
    void dotBlock(int start, int end, PatchData data);
}
