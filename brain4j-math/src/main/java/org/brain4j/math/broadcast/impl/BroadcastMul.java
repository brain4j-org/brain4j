package org.brain4j.math.broadcast.impl;

import org.brain4j.math.Tensors;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.broadcast.BroadcastOperation;

import java.util.Arrays;

public class BroadcastMul implements BroadcastOperation {
    
    /**
     * Checks if a tensor has standard contiguous strides (row-major order).
     */
    private boolean isContiguous(Tensor tensor) {
        int[] shape = tensor.shape();
        int[] strides = tensor.strides();
        int[] expectedStrides = Tensors.computeStrides(shape);
        return Arrays.equals(strides, expectedStrides);
    }

    @Override
    public Tensor defaultOp(Tensor A, Tensor B) {
        int[] shape = A.shape();
        int[] otherShape = B.shape();

        float[] aData = A.data();
        float[] bData = B.data();

        if (Arrays.equals(shape, otherShape) && isContiguous(A) && isContiguous(B)) {
            for (int i = 0; i < aData.length; i++) {
                aData[i] *= bData[i];
            }
            return A;
        }

        if (Arrays.equals(shape, otherShape)) {
            int[] stridesA = A.strides();
            int[] stridesB = B.strides();
            int rank = shape.length;
            int total = A.elements();
            int[] index = new int[rank];

            for (int i = 0; i < total; i++) {
                unravelIndex(i, shape, index);

                int aIdx = 0;
                int bIdx = 0;
                for (int d = 0; d < rank; d++) {
                    aIdx += index[d] * stridesA[d];
                    bIdx += index[d] * stridesB[d];
                }

                aData[aIdx] *= bData[bIdx];
            }
            return A;
        }

        if (shape.length == 2 && otherShape.length == 1 && shape[1] == otherShape[0] && isContiguous(A)) {
            int batch = shape[0];
            int dimension = shape[1];

            for (int i = 0; i < batch; i++) {
                int base = i * dimension;

                for (int j = 0; j < dimension; j++) {
                    aData[base + j] *= bData[j];
                }
            }

            return A;
        }
        
        if (shape.length == 3 && otherShape.length == 1 && shape[2] == otherShape[0] && isContiguous(A)) {
            int d0 = shape[0];
            int d1 = shape[1];
            int d2 = shape[2];
            
            int total = d0 * d1 * d2;
            
            for (int idx = 0; idx < total; idx++) {
                int k = idx % d2;
                aData[idx] *= bData[k];
            }
            
            return A;
        }
        
        return fallbackOp(A, B);
    }

    @Override
    public Tensor fallbackOp(Tensor A, Tensor B) {
        int[] shapeA = A.shape();
        int[] shapeB = B.shape();

        float[] aData = A.data();
        float[] bData = B.data();

        int[] broadcastedShape = broadcastShape(shapeA, shapeB);

        if (!Arrays.equals(shapeA, broadcastedShape)) {
            throw new IllegalArgumentException("Broadcast result does not match shape of A");
        }

        int total = A.elements();

        int[] stridesA = A.strides();
        int[] stridesB = B.strides();
        int[] index = new int[shapeA.length];

        for (int i = 0; i < total; i++) {
            unravelIndex(i, shapeA, index);

            int aIndex = 0;
            int bIndex = 0;

            for (int d = 0; d < shapeA.length; d++) {
                aIndex += index[d] * stridesA[d];
                int dimB = (shapeB.length - shapeA.length + d);
                if (dimB >= 0 && shapeB[dimB] != 1) {
                    bIndex += index[d] * stridesB[dimB];
                }
            }

            aData[aIndex] *= bData[bIndex];
        }

        return A;
    }
}
