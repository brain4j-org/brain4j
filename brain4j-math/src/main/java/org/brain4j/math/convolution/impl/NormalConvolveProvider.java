package org.brain4j.math.convolution.impl;

import org.brain4j.math.convolution.ConvolveProvider;

public class NormalConvolveProvider implements ConvolveProvider {
    
    @Override
    public void dotBlock(int start, int end, PatchData data) {
        int patchSize = data.patchSize();
        int filterOffset = data.filterOffset();
        int outOffset = data.outOffset();
        
        float[] filterData = data.filter();
        float[] patchData = data.patch();
        float[] outData = data.out();
        
        for (int p = start; p < end; p++) {
            float sum = 0f;
            int rowOffset = p * patchSize;
            
            for (int i = 0; i < patchSize; i++) {
                sum += filterData[filterOffset + i] * patchData[rowOffset + i];
            }
            
            outData[outOffset + p] = sum;
        }
    }
}
