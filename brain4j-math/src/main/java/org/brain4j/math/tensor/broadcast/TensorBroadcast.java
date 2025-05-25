package org.brain4j.math.tensor.broadcast;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;
import org.brain4j.math.tensor.impl.TensorImplBase;

public class TensorBroadcast {

    private static int[] padStrides(int[] strides, int L) {
        int pad = L - strides.length;
        int[] out = new int[L];

        System.arraycopy(strides, 0, out, pad, strides.length);
        return out;
    }

    private static int[] padShape(int[] shape, int L) {
        int[] padded = new int[L];
        int diff = L - shape.length;

        for (int i = 0; i < diff; i++) {
            padded[i] = 1;
        }

        System.arraycopy(shape, 0, padded, diff, shape.length);
        return padded;
    }

    public static Tensor add(Tensor A, Tensor B) {
        float[] dataA = A.data(), dataB = B.data();
        int[] shapeA = A.shape(), shapeB = B.shape();
        int[] strideA = A.strides(), strideB = B.strides();

        int L = Math.max(shapeA.length, shapeB.length);
        int[] padA = padShape(shapeA, L);
        int[] padB = padShape(shapeB, L);
        int[] stridePA = padStrides(strideA, L);
        int[] stridePB = padStrides(strideB, L);

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

        int total = C.elements();
        int[] coord = new int[L];

        int offA = 0, offB = 0;

        for (int idx = 0; idx < total; idx++) {
            dataC[idx] = dataA[offA] + dataB[offB];

            for (int dim = L - 1; dim >= 0; dim--) {
                coord[dim]++;
                if (coord[dim] < shapeC[dim]) {
                    offA += (padA[dim] == 1 ? 0 : stridePA[dim]);
                    offB += (padB[dim] == 1 ? 0 : stridePB[dim]);
                    break;
                } else {
                    coord[dim] = 0;
                    offA -= (padA[dim] == 1 ? 0 : stridePA[dim] * (shapeC[dim] - 1));
                    offB -= (padB[dim] == 1 ? 0 : stridePB[dim] * (shapeC[dim] - 1));
                }
            }
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
        int[] stridePA = padStrides(strideA, L);
        int[] stridePB = padStrides(strideB, L);

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

        int total = C.elements();
        int[] coord = new int[L];

        int offA = 0, offB = 0;

        for (int idx = 0; idx < total; idx++) {
            dataC[idx] = dataA[offA] - dataB[offB];

            for (int dim = L - 1; dim >= 0; dim--) {
                coord[dim]++;
                if (coord[dim] < shapeC[dim]) {
                    offA += (padA[dim] == 1 ? 0 : stridePA[dim]);
                    offB += (padB[dim] == 1 ? 0 : stridePB[dim]);
                    break;
                } else {
                    coord[dim] = 0;
                    offA -= (padA[dim] == 1 ? 0 : stridePA[dim] * (shapeC[dim] - 1));
                    offB -= (padB[dim] == 1 ? 0 : stridePB[dim] * (shapeC[dim] - 1));
                }
            }
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
        int[] stridePA = padStrides(strideA, L);
        int[] stridePB = padStrides(strideB, L);

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

        int total = C.elements();
        int[] coord = new int[L];

        int offA = 0, offB = 0;

        for (int idx = 0; idx < total; idx++) {
            dataC[idx] = dataA[offA] * dataB[offB];

            for (int dim = L - 1; dim >= 0; dim--) {
                coord[dim]++;
                if (coord[dim] < shapeC[dim]) {
                    offA += (padA[dim] == 1 ? 0 : stridePA[dim]);
                    offB += (padB[dim] == 1 ? 0 : stridePB[dim]);
                    break;
                } else {
                    coord[dim] = 0;
                    offA -= (padA[dim] == 1 ? 0 : stridePA[dim] * (shapeC[dim] - 1));
                    offB -= (padB[dim] == 1 ? 0 : stridePB[dim] * (shapeC[dim] - 1));
                }
            }
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
        int[] stridePA = padStrides(strideA, L);
        int[] stridePB = padStrides(strideB, L);

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

        int total = C.elements();
        int[] coord = new int[L];

        int offA = 0, offB = 0;

        for (int idx = 0; idx < total; idx++) {
            dataC[idx] = dataA[offA] / dataB[offB];

            for (int dim = L - 1; dim >= 0; dim--) {
                coord[dim]++;
                if (coord[dim] < shapeC[dim]) {
                    offA += (padA[dim] == 1 ? 0 : stridePA[dim]);
                    offB += (padB[dim] == 1 ? 0 : stridePB[dim]);
                    break;
                } else {
                    coord[dim] = 0;
                    offA -= (padA[dim] == 1 ? 0 : stridePA[dim] * (shapeC[dim] - 1));
                    offB -= (padB[dim] == 1 ? 0 : stridePB[dim] * (shapeC[dim] - 1));
                }
            }
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
        int[] stridePA = padStrides(strideA, L);
        int[] stridePB = padStrides(strideB, L);

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

        int total = C.elements();
        int[] coord = new int[L];

        int offA = 0, offB = 0;

        for (int idx = 0; idx < total; idx++) {
            dataC[idx] = (float) Math.pow(dataA[offA], dataB[offB]);

            for (int dim = L - 1; dim >= 0; dim--) {
                coord[dim]++;
                if (coord[dim] < shapeC[dim]) {
                    offA += (padA[dim] == 1 ? 0 : stridePA[dim]);
                    offB += (padB[dim] == 1 ? 0 : stridePB[dim]);
                    break;
                } else {
                    coord[dim] = 0;
                    offA -= (padA[dim] == 1 ? 0 : stridePA[dim] * (shapeC[dim] - 1));
                    offB -= (padB[dim] == 1 ? 0 : stridePB[dim] * (shapeC[dim] - 1));
                }
            }
        }

        return C;
    }
}
