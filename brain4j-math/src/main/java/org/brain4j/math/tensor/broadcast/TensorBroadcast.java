package org.brain4j.math.tensor.broadcast;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;
import org.brain4j.math.tensor.impl.TensorImplBase;

public class TensorBroadcast {

    private static int[] padShape(int[] shape, int L) {
        int[] padded = new int[L];
        int diff = L - shape.length;

        for (int i = 0; i < diff; i++) {
            padded[i] = 1;
        }

        System.arraycopy(shape, 0, padded, diff, shape.length);
        return padded;
    }

    private static int[] padStrides(int[] strides, int L) {
        int pad = L - strides.length;
        int[] out = new int[L];

        System.arraycopy(strides, 0, out, pad, strides.length);
        return out;
    }

    public static Tensor add(Tensor A, Tensor B) {
        float[] dataA = A.data(), dataB = B.data();
        int[] shapeA = A.shape(), shapeB = B.shape();
        int[] strideA = A.strides(), strideB = B.strides();

        int L = Math.max(shapeA.length, shapeB.length);

        int[] padA = padShape(shapeA, L);
        int[] padB = padShape(shapeB, L);

        int[] shapeC = new int[L];

        for (int i = 0; i < L; i++) {
            int a = padA[i], b = padB[i];

            if (a == b || a == 1 || b == 1) {
                shapeC[i] = Math.max(a, b);
            } else {
                throw new IllegalArgumentException("Shape mismatch in dimension " + i + ": " + a + " vs " + b);
            }
        }

        Tensor C = Tensors.zeros(shapeC);
        float[] dataC = C.data();
        int[] strideC = C.strides();
        int total = C.elements();

        int[] padStrideA = padStrides(strideA, L);
        int[] padStrideB = padStrides(strideB, L);

        for (int idx = 0; idx < total; idx++) {
            int rem = idx;
            int offA = 0, offB = 0;

            for (int i = 0; i < L; i++) {
                int coord = rem / strideC[i];
                rem %= strideC[i];

                int coordA = (padA[i] == 1) ? 0 : coord;
                int coordB = (padB[i] == 1) ? 0 : coord;

                offA += coordA * padStrideA[i];
                offB += coordB * padStrideB[i];
            }

            dataC[idx] = dataA[offA] + dataB[offB];
        }

        return C;
    }

    public static Tensor sub(Tensor A, Tensor B) {
        float[] dataA = A.data(), dataB = B.data();
        int[] shapeA = A.shape(), shapeB = B.shape();
        int[] strideA = A.strides(), strideB = B.strides();

        int L = Math.max(shapeA.length, shapeB.length);

        int[] padA = padShape(shapeA, L);
        int[] padB = padShape(shapeB, L);

        int[] shapeC = new int[L];

        for (int i = 0; i < L; i++) {
            int a = padA[i], b = padB[i];

            if (a == b || a == 1 || b == 1) {
                shapeC[i] = Math.max(a, b);
            } else {
                throw new IllegalArgumentException("Shape mismatch in dimension " + i + ": " + a + " vs " + b);
            }
        }

        Tensor C = Tensors.zeros(shapeC);
        float[] dataC = C.data();
        int[] strideC = C.strides();
        int total = C.elements();

        int[] padStrideA = padStrides(strideA, L);
        int[] padStrideB = padStrides(strideB, L);

        for (int idx = 0; idx < total; idx++) {
            int rem = idx;
            int offA = 0, offB = 0;

            for (int i = 0; i < L; i++) {
                int coord = rem / strideC[i];
                rem %= strideC[i];

                int coordA = (padA[i] == 1) ? 0 : coord;
                int coordB = (padB[i] == 1) ? 0 : coord;

                offA += coordA * padStrideA[i];
                offB += coordB * padStrideB[i];
            }

            dataC[idx] = dataA[offA] - dataB[offB];
        }

        return C;
    }

    public static Tensor mul(Tensor A, Tensor B) {
        float[] dataA = A.data(), dataB = B.data();
        int[] shapeA = A.shape(), shapeB = B.shape();
        int[] strideA = A.strides(), strideB = B.strides();

        int L = Math.max(shapeA.length, shapeB.length);

        int[] padA = padShape(shapeA, L);
        int[] padB = padShape(shapeB, L);

        int[] shapeC = new int[L];

        for (int i = 0; i < L; i++) {
            int a = padA[i], b = padB[i];

            if (a == b || a == 1 || b == 1) {
                shapeC[i] = Math.max(a, b);
            } else {
                throw new IllegalArgumentException("Shape mismatch in dimension " + i + ": " + a + " vs " + b);
            }
        }

        Tensor C = Tensors.zeros(shapeC);
        float[] dataC = C.data();
        int[] strideC = C.strides();
        int total = C.elements();

        int[] padStrideA = padStrides(strideA, L);
        int[] padStrideB = padStrides(strideB, L);

        for (int idx = 0; idx < total; idx++) {
            int rem = idx;
            int offA = 0, offB = 0;

            for (int i = 0; i < L; i++) {
                int coord = rem / strideC[i];
                rem %= strideC[i];

                int coordA = (padA[i] == 1) ? 0 : coord;
                int coordB = (padB[i] == 1) ? 0 : coord;

                offA += coordA * padStrideA[i];
                offB += coordB * padStrideB[i];
            }

            dataC[idx] = dataA[offA] * dataB[offB];
        }

        return C;
    }

    public static Tensor div(Tensor A, Tensor B) {
        float[] dataA = A.data(), dataB = B.data();
        int[] shapeA = A.shape(), shapeB = B.shape();
        int[] strideA = A.strides(), strideB = B.strides();

        int L = Math.max(shapeA.length, shapeB.length);

        int[] padA = padShape(shapeA, L);
        int[] padB = padShape(shapeB, L);

        int[] shapeC = new int[L];

        for (int i = 0; i < L; i++) {
            int a = padA[i], b = padB[i];

            if (a == b || a == 1 || b == 1) {
                shapeC[i] = Math.max(a, b);
            } else {
                throw new IllegalArgumentException("Shape mismatch in dimension " + i + ": " + a + " vs " + b);
            }
        }

        Tensor C = Tensors.zeros(shapeC);
        float[] dataC = C.data();
        int[] strideC = C.strides();
        int total = C.elements();

        int[] padStrideA = padStrides(strideA, L);
        int[] padStrideB = padStrides(strideB, L);

        for (int idx = 0; idx < total; idx++) {
            int rem = idx;
            int offA = 0, offB = 0;

            for (int i = 0; i < L; i++) {
                int coord = rem / strideC[i];
                rem %= strideC[i];

                int coordA = (padA[i] == 1) ? 0 : coord;
                int coordB = (padB[i] == 1) ? 0 : coord;

                offA += coordA * padStrideA[i];
                offB += coordB * padStrideB[i];
            }

            dataC[idx] = dataA[offA] / dataB[offB];
        }

        return C;
    }

    public static Tensor pow(Tensor A, Tensor B) {
        float[] dataA = A.data(), dataB = B.data();
        int[] shapeA = A.shape(), shapeB = B.shape();
        int[] strideA = A.strides(), strideB = B.strides();

        int L = Math.max(shapeA.length, shapeB.length);

        int[] padA = padShape(shapeA, L);
        int[] padB = padShape(shapeB, L);

        int[] shapeC = new int[L];

        for (int i = 0; i < L; i++) {
            int a = padA[i], b = padB[i];

            if (a == b || a == 1 || b == 1) {
                shapeC[i] = Math.max(a, b);
            } else {
                throw new IllegalArgumentException("Shape mismatch in dimension " + i + ": " + a + " vs " + b);
            }
        }

        Tensor C = Tensors.zeros(shapeC);
        float[] dataC = C.data();
        int[] strideC = C.strides();
        int total = C.elements();

        int[] padStrideA = padStrides(strideA, L);
        int[] padStrideB = padStrides(strideB, L);

        for (int idx = 0; idx < total; idx++) {
            int rem = idx;
            int offA = 0, offB = 0;

            for (int i = 0; i < L; i++) {
                int coord = rem / strideC[i];
                rem %= strideC[i];

                int coordA = (padA[i] == 1) ? 0 : coord;
                int coordB = (padB[i] == 1) ? 0 : coord;

                offA += coordA * padStrideA[i];
                offB += coordB * padStrideB[i];
            }

            dataC[idx] = (float) Math.pow(dataA[offA], dataB[offB]);
        }

        return C;
    }
}
