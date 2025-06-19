package org.brain4j.math.tensor.cpu.matmul;

import java.util.concurrent.ForkJoinPool;

public interface MatmulProvider {

    void multiply(
        ForkJoinPool pool,
        int batch,
        int m, int n, int p,
        float[] A, float[] B, float[] C,
        int batchA, int batchB
    );
}
