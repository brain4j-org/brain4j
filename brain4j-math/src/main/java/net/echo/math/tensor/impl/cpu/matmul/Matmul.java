package net.echo.math.tensor.impl.cpu.matmul;

import java.util.concurrent.ForkJoinPool;

public interface Matmul {

    void multiply(
            int batch, int m, int n, int p, float[] A, float[] B, float[] C, ForkJoinPool pool
    );

}
