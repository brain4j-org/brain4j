package org.brain4j.math.operations.sum.impl;

import org.brain4j.math.Tensors;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.operations.sum.TensorReducer;

import java.util.Arrays;
import java.util.stream.IntStream;

public class ScalarTensorReducer implements TensorReducer {
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
        
        int finalInnerSize = innerSize;
        IntStream.range(0, outerSize).parallel().forEach(outer -> {
            int baseOuter = outer * reducedSize * finalInnerSize;
            for (int i = 0; i < reducedSize; i++) {
                int base = baseOuter + i * finalInnerSize;
                for (int inner = 0; inner < finalInnerSize; inner++) {
                    resultData[outer * finalInnerSize + inner] += data[base + inner];
                }
            }
        });
        
        return result;
    }
}
