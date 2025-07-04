package org.brain4j.common.tensor.matmul;

public record MatmulParameters(
    float[] A, float[] B, float[] C,
    int m, int n, int p,
    boolean transpose
) {
    public int mn() {
        return m * n;
    }

    public int np() {
        return n * p;
    }

    public int mp() {
        return m * p;
    }
}
