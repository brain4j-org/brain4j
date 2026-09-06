package org.brain4j.math.broadcast.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.broadcast.BroadcastOperation;

import java.util.Arrays;

public class BroadcastAdd implements BroadcastOperation {

    @Override
    public Tensor defaultOp(Tensor A, Tensor B) {
        int[] shapeA = A.shape();
        int[] shapeB = B.shape();

        float[] aData = A.data();
        float[] bData = B.data();

        if (Arrays.equals(shapeA, shapeB)) {
            for (int i = 0; i < aData.length; i++) {
                aData[i] += bData[i];
            }

            return A;
        }

        if (shapeA.length == 2 && shapeB.length == 1 && shapeA[1] == shapeB[0]) {
            int batch = shapeA[0];
            int dimension = shapeA[1];

            int total = batch * dimension;
            for (int idx = 0; idx < total; idx++) {
                int j = idx % dimension;
                aData[idx] += bData[j];
            }

            return A;
        }

        if (shapeA.length == 3) {
            int d0 = shapeA[0]; // a
            int d1 = shapeA[1]; // b
            int d2 = shapeA[2]; // c

            int total = d0 * d1 * d2;

            // [a, b, c] + [c]
            if (shapeB.length == 1 && shapeA[2] == shapeB[0]) {
                for (int idx = 0; idx < total; idx++) {
                    int k = idx % d2;
                    aData[idx] += bData[k];
                }

                return A;
            }

            // [a, b, c] + [b, c]
            if (shapeB.length == 2 && shapeA[1] == shapeB[0] && shapeA[2] == shapeB[1]) {
                int stride = d1 * d2; // size of [b, c]

                for (int batch = 0; batch < d0; batch++) {
                    int offset = batch * stride;
                    for (int i = 0; i < stride; i++) {
                        aData[offset + i] += bData[i];
                    }
                }

                return A;
            }
        }

        // optimized version for convolutions
        if (isBiasShape(shapeA, shapeB)) {
            addBiasInPlace(A, B);
            return A;
        }

        return fallbackOp(A, B);
    }

    private void addBiasInPlace(Tensor output, Tensor bias) {
        int[] shape = output.shape();
        float[] outData = output.data();
        float[] biasData = bias.data();

        int batch = shape[0];
        int filters = shape[1];
        int height = shape[2];
        int width  = shape[3];

        int hw = height * width;
        int strideB = filters * hw;

        for (int b = 0; b < batch; b++) {
            int baseB = b * strideB;

            for (int f = 0; f < filters; f++) {
                int baseF = baseB + f * hw;
                float biasVal = biasData[f];

                for (int i = 0; i < hw; i++) {
                    outData[baseF + i] += biasVal;
                }
            }
        }
    }

    private boolean isBiasShape(int[] a, int[] b) {
        if (a.length != 4) return false; // [B, F, H, W]
        if (b.length == 1 && b[0] == a[1]) return true;
        return b.length == 4 && b[0] == 1 && b[2] == 1 && b[3] == 1 && b[1] == a[1];
    }

    @Override
    public Tensor fallbackOp(Tensor A, Tensor B) {
        int[] effStrideB = makeStrideMap(A, B);
        int[] shapeA = A.shape();

        float[] aData = A.data();
        float[] bData = B.data();

        int rankA = shapeA.length;
        int bIndex = 0;
        int[] indexA = new int[rankA];

        int total = A.elements();

        for (int i = 0; i < total; i++) {
            aData[i] += bData[bIndex];

            for (int d = rankA - 1; ; d--) {
                if (++indexA[d] < shapeA[d]) {
                    bIndex += effStrideB[d];
                    break;
                }
                indexA[d] = 0;
                bIndex -= effStrideB[d] * (shapeA[d] - 1);

                if (d == 0) break;
            }
        }

        return A;
    }
}
