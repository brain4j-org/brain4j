package org.brain4j.math.broadcast;

import org.brain4j.math.tensor.Tensor;

import java.util.Arrays;

/**
 * Interface defining a broadcastable-operation
 * @author xEcho1337
 */
public interface BroadcastOperation {

    /**
     * Computes the strides map used to broadcast B on A.
     * @param A the first tensor
     * @param B the second tensor (the one to broadcast)
     * @return a new int array which holds the strides for the broadcasting
     */
    default int[] makeStrideMap(Tensor A, Tensor B) {
        int[] shapeA = A.shape();
        int[] shapeB = B.shape();
        int[] stridesB = B.strides();

        int rankA = shapeA.length;
        int rankB = shapeB.length;
        int dimOffset = rankB - rankA;

        int[] effStrideB = new int[rankA];

        for (int d = 0; d < rankA; d++) {
            int dimB = dimOffset + d;
            if (dimB < 0) {
                effStrideB[d] = 0;
                continue;
            }

            if (shapeB[dimB] == 1) {
                effStrideB[d] = 0;
            } else {
                effStrideB[d] = stridesB[dimB];
            }
        }

        return effStrideB;
    }

    /**
     * Default method called when broadcasting two tensors.
     * @param A the first tensor
     * @param B the second tensor
     * @return A tensor resulting from A combination of the two inputs
     */
    Tensor defaultOp(Tensor A, Tensor B);

    /**
     * Fallback method if {@link BroadcastOperation#defaultOp(Tensor, Tensor)} does not support the broadcasting.
     * @param A the first tensor
     * @param B the second tensor
     * @return A tensor resulting from A combination of the two inputs
     */
    Tensor fallbackOp(Tensor A, Tensor B);

    default void unravelIndex(int flatIndex, int[] shape, int[] result) {
        for (int i = shape.length - 1; i >= 0; i--) {
            result[i] = flatIndex % shape[i];
            flatIndex /= shape[i];
        }
    }

    default int[] broadcastShape(int[] a, int[] b) {
        int len = Math.max(a.length, b.length);
        int[] result = new int[len];

        for (int i = 0; i < len; i++) {
            int dimA = i >= len - a.length ? a[i - (len - a.length)] : 1;
            int dimB = i >= len - b.length ? b[i - (len - b.length)] : 1;

            if (dimA != dimB && dimA != 1 && dimB != 1) {
                throw new IllegalArgumentException("Incompatible shapes for broadcasting: " +
                    Arrays.toString(a) + " vs " + Arrays.toString(b));
            }

            result[i] = Math.max(dimA, dimB);
        }

        return result;
    }
}
