package org.brain4j.math.operations.sum.impl;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import org.brain4j.math.Tensors;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.operations.sum.TensorReducer;

import java.util.Arrays;

public class SIMDTensorReducer implements TensorReducer {
    
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    
    @Override
    public Tensor sum(Tensor tensor, int dim, boolean keepDim) {
        int[] shape = tensor.shape();
        
        if (dim < 0 || dim >= shape.length) {
            throw new IllegalArgumentException("Dimension " + dim + " out of bounds for tensor of shape " + Arrays.toString(shape));
        }
        
        int[] newShape = Tensors.computeNewShape(shape, dim, keepDim);
        int reducedSize = shape[dim];
        
        Tensor result = Tensors.zeros(newShape);
        float[] resultData = result.data();
        float[] data = tensor.data();
        
        int outerSize = 1;
        int innerSize = 1;
        
        for (int i = 0; i < dim; i++) outerSize *= shape[i];
        for (int i = dim + 1; i < shape.length; i++) innerSize *= shape[i];
        
        for (int outer = 0; outer < outerSize; outer++) {
            int baseOuter = outer * reducedSize * innerSize;
            for (int i = 0; i < reducedSize; i++) {
                int base = baseOuter + i * innerSize;
                
                int inner = 0;
                for (; inner + SPECIES.length() <= innerSize; inner += SPECIES.length()) {
                    var acc = FloatVector.fromArray(SPECIES, resultData, outer * innerSize + inner);
                    var v = FloatVector.fromArray(SPECIES, data, base + inner);
                    acc = acc.add(v);
                    acc.intoArray(resultData, outer * innerSize + inner);
                }
                
                for (; inner < innerSize; inner++) {
                    resultData[outer * innerSize + inner] += data[base + inner];
                }
            }
        }
        
        return result;
    }
}
