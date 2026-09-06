package org.brain4j.math.convolution.impl;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.brain4j.math.convolution.ConvolveProvider;

public class SIMDConvolveProvider implements ConvolveProvider {
    
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    @Override
    public void dotBlock(int start, int end, PatchData data) {
        int patchSize = data.patchSize();
        int filterOffset = data.filterOffset();
        int outOffset = data.outOffset();

        float[] filterData = data.filter();
        float[] patchData = data.patch();
        float[] outData = data.out();

        for (int p = start; p < end; p++) {
            int rowOffset = p * patchSize;
            
            int i = 0;
            int loopBound = SPECIES.loopBound(patchSize);
            var register = FloatVector.zero(SPECIES);
            
            for (; i < loopBound; i += SPECIES.length()) {
                var v1 = FloatVector.fromArray(SPECIES, filterData, filterOffset + i);
                var v2 = FloatVector.fromArray(SPECIES, patchData, rowOffset + i);
                register = register.add(v1.mul(v2));
            }
            
            float sum = register.reduceLanes(VectorOperators.ADD);
            
            for (; i < patchSize; i++) {
                sum += filterData[filterOffset + i] * patchData[rowOffset + i];
            }
            
            outData[outOffset + p] = sum;
        }
    }
}
