package net.echo.math4j.math.tensor.impl;

import net.echo.math4j.math.tensor.Tensor;

public class TensorNative extends TensorCPU {

    static {
        System.load("/Users/echo/IdeaProjects/brain4j/brain4j-core/src/test/java/backend/libs/libtensor_backend.dylib");
    }

    public TensorNative(int... shape) {
        super(shape);
    }

    public native float[] matmul(float[] a, float[] b, int m, int n, int p);

    @Override
    public Tensor matmul(Tensor other) {
        int[] shape = this.shape();
        int[] otherShape = other.shape();

        int m = shape[0];
        int n = shape[1];
        int p = otherShape[1];

        if (n != otherShape[0]) {
            throw new IllegalArgumentException("Dimensions do not match: " + n + " != " + otherShape[0]);
        }

        float[] a = this.getData();
        float[] b = other.getData();

        float[] result = matmul(a, b, m, n, p);

        TensorNative tensor = new TensorNative(shape());
        tensor.data = result;

        return tensor;
    }
}
