package org.brain4j.math.broadcast;

import org.brain4j.math.gpu.device.DeviceUtils;
import org.brain4j.math.tensor.Tensor;

import java.util.Arrays;

public abstract class BaseBroadcast implements BroadcastOperation {

    protected final boolean vector = DeviceUtils.isSimdAvailable();

    @Override
    public final Tensor defaultOp(Tensor A, Tensor B) {
        int[] shapeA = A.shape();
        int[] shapeB = B.shape();

        float[] aData = A.data();
        float[] bData = B.data();

        if (Arrays.equals(shapeA, shapeB)) {
            if (vector) {
                sameShapeSimd(aData, bData);
            } else {
                sameShape(aData, bData);
            }
            return A;
        }

        if (shapeA.length == 2 && shapeB.length == 1 && shapeA[1] == shapeB[0]) {
            if (vector) {
                rowWiseSimd(aData, bData, shapeA[0], shapeA[1]);
            } else {
                rowWise(aData, bData, shapeA[0], shapeA[1]);
            }
            return A;
        }

        if (shapeA.length == 3) {
            int d0 = shapeA[0];
            int d1 = shapeA[1];
            int d2 = shapeA[2];
            int total = d0 * d1 * d2;

            // [a, b, c] op [c]
            if (shapeB.length == 1 && shapeA[2] == shapeB[0]) {
                if (vector) {
                    rowWiseSimd(aData, bData, total / d2, d2);
                } else {
                    rowWise(aData, bData, total / d2, d2);
                }
                return A;
            }

            // [a, b, c] op [b, c]
            if (shapeB.length == 2 && shapeA[1] == shapeB[0] && shapeA[2] == shapeB[1]) {
                if (vector) {
                    rowWiseSimd(aData, bData, d0, d1 * d2);
                } else {
                    rowWise(aData, bData, d0, d1 * d2);
                }
                return A;
            }
        }

        // [B, F, H, W] op [F]
        if (isBiasShape(shapeA, shapeB)) {
            if (vector) {
                biasSimd(A, B);
            } else {
                bias(A, B);
            }
            return A;
        }

        return fallbackOp(A, B);
    }

    protected abstract void sameShape(float[] a, float[] b);

    protected abstract void sameShapeSimd(float[] a, float[] b);

    protected abstract void rowWise(float[] a, float[] b, int batch, int dim);

    protected abstract void rowWiseSimd(float[] a, float[] b, int batch, int dim);

    protected abstract void bias(Tensor output, Tensor biasTensor);

    protected abstract void biasSimd(Tensor output, Tensor biasTensor);

    @Override
    public Tensor fallbackOp(Tensor A, Tensor B) {
        int[] shapeA = A.shape();
        int[] shapeB = B.shape();

        float[] aData = A.data();
        float[] bData = B.data();

        int[] broadcastedShape = broadcastShape(shapeA, shapeB);
        if (!Arrays.equals(shapeA, broadcastedShape)) {
            throw new IllegalArgumentException("Broadcast result does not match shape of A! A = " +
                Arrays.toString(shapeA) + ", B = " + Arrays.toString(shapeB));
        }

        int total = A.elements();
        int[] stridesB = B.strides();
        int[] index = new int[shapeA.length];

        for (int i = 0; i < total; i++) {
            unravelIndex(i, shapeA, index);
            int bIndex = 0;
            for (int d = 0; d < shapeA.length; d++) {
                int dimB = shapeB.length - shapeA.length + d;
                if (dimB >= 0 && shapeB[dimB] != 1) {
                    bIndex += index[d] * stridesB[dimB];
                }
            }
            aData[i] = scalar(aData[i], bData[bIndex]);
        }

        return A;
    }

    protected abstract float scalar(float a, float b);
}
