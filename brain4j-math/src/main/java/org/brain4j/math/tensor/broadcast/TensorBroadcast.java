package org.brain4j.math.tensor.broadcast;

import org.brain4j.math.tensor.Tensor;

import java.util.Arrays;

public class TensorBroadcast {

    public static Tensor add(Tensor A, Tensor B) {
        int[] shape = A.shape();
        int[] otherShape = B.shape();

        float[] aData = A.data();
        float[] bData = B.data();

        if (Arrays.equals(shape, otherShape)) {
            for (int i = 0; i < aData.length; i++) {
                aData[i] += bData[i];
            }

            return A;
        }

        if (shape.length == 2 && otherShape.length == 1 && shape[1] == otherShape[0]) {
            int batch = shape[0];
            int dimension = shape[1];

            for (int i = 0; i < batch; i++) {
                int base = i * dimension;

                for (int j = 0; j < dimension; j++) {
                    aData[base + j] += bData[j];
                }
            }

            return A;
        }

        throw new IllegalArgumentException(
            "Cannot broadcast shapes " + Arrays.toString(shape) + " and " + Arrays.toString(otherShape)
        );
    }

    public static Tensor sub(Tensor A, Tensor B) {
        int[] shape = A.shape();
        int[] otherShape = B.shape();

        float[] aData = A.data();
        float[] bData = B.data();

        if (Arrays.equals(shape, otherShape)) {
            for (int i = 0; i < aData.length; i++) {
                aData[i] -= bData[i];
            }

            return A;
        }

        if (shape.length == 2 && otherShape.length == 1 && shape[1] == otherShape[0]) {
            int batch = shape[0];
            int dimension = shape[1];

            for (int i = 0; i < batch; i++) {
                int base = i * dimension;

                for (int j = 0; j < dimension; j++) {
                    aData[base + j] -= bData[j];
                }
            }

            return A;
        }

        throw new IllegalArgumentException(
            "Cannot broadcast shapes " + Arrays.toString(shape) + " and " + Arrays.toString(otherShape)
        );
    }

    public static Tensor mul(Tensor A, Tensor B) {
        int[] shape = A.shape();
        int[] otherShape = B.shape();

        float[] aData = A.data();
        float[] bData = B.data();

        if (Arrays.equals(shape, otherShape)) {
            for (int i = 0; i < aData.length; i++) {
                aData[i] *= bData[i];
            }

            return A;
        }

        if (shape.length == 2 && otherShape.length == 1 && shape[1] == otherShape[0]) {
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

        throw new IllegalArgumentException(
            "Cannot broadcast shapes " + Arrays.toString(shape) + " and " + Arrays.toString(otherShape)
        );
    }

    public static Tensor div(Tensor A, Tensor B) {
        int[] shape = A.shape();
        int[] otherShape = B.shape();

        float[] aData = A.data();
        float[] bData = B.data();

        if (Arrays.equals(shape, otherShape)) {
            for (int i = 0; i < aData.length; i++) {
                aData[i] /= bData[i];
            }

            return A;
        }

        if (shape.length == 2 && otherShape.length == 1 && shape[1] == otherShape[0]) {
            int batch = shape[0];
            int dimension = shape[1];

            for (int i = 0; i < batch; i++) {
                int base = i * dimension;

                for (int j = 0; j < dimension; j++) {
                    aData[base + j] /= bData[j];
                }
            }

            return A;
        }

        throw new IllegalArgumentException(
            "Cannot broadcast shapes " + Arrays.toString(shape) + " and " + Arrays.toString(otherShape)
        );
    }

    public static Tensor pow(Tensor A, Tensor B) {
        int[] shape = A.shape();
        int[] otherShape = B.shape();

        float[] aData = A.data();
        float[] bData = B.data();

        if (Arrays.equals(shape, otherShape)) {
            for (int i = 0; i < aData.length; i++) {
                aData[i] = (float) Math.pow(aData[i], bData[i]);
            }

            return A;
        }

        if (shape.length == 2 && otherShape.length == 1 && shape[1] == otherShape[0]) {
            int batch = shape[0];
            int dimension = shape[1];

            for (int i = 0; i < batch; i++) {
                int base = i * dimension;

                for (int j = 0; j < dimension; j++) {
                    aData[base + j] = (float) Math.pow(aData[base + j], bData[j]);
                }
            }

            return A;
        }

        throw new IllegalArgumentException(
            "Cannot broadcast shapes " + Arrays.toString(shape) + " and " + Arrays.toString(otherShape)
        );
    }
}
