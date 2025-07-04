package org.brain4j.common.tensor.matmul.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.matmul.MatmulParameters;
import org.brain4j.common.tensor.matmul.MatmulProvider;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public class NormalMatmulProvider implements MatmulProvider {

    private static final int WORK_THRESHOLD = 1;
    private static final int COMPLEXITY_THRESHOLD = 65536;

    private static boolean isOverThreshold(int work, int np) {
        return work > WORK_THRESHOLD && work * np > COMPLEXITY_THRESHOLD;
    }

    public void multiply(
        int batch, int m, int n, int p,
        boolean transpose,
        float[] A, float[] B, float[] C,
        ForkJoinPool pool
    ) {
        int start = 0;
        int end = batch * m;
        int mn = m * n;
        int np = n * p;
        int mp = m * p;

        int work = end - start;

        if (!isOverThreshold(work, np)) {
            matmulBlock(A, B, C, start, end, transpose, m, n, p, mn, np, mp);
            return;
        }

        MatmulParameters parameters = new MatmulParameters(A, B, C, m, n, p, transpose);
        ScalarAction action = new ScalarAction(start, end, parameters);

        pool.invoke(action);
    }

    @Override
    public void multiply(ForkJoinPool pool, Tensor a, Tensor b, Tensor c) {
        float[] A = a.data();
        float[] B = b.data();
        float[] C = c.data();

        int rankA = a.rank();
        int rankB = b.rank();

        int[] shapeA = a.shape();
        int[] shapeB = b.shape();
        int[] shapeC = c.shape();

        int batch = 1;
        for (int d : shapeC) batch *= d;

        int m = shapeA[rankA - 2];
        int n = shapeA[rankA - 1];
        int p = shapeB[rankB - 1];

        multiply(batch, m, n, p, b.transposed(), A, B, C, pool);
    }

    private void matmulBlock(
        float[] A, float[] B, float[] C,
        int start, int end,
        boolean transpose,
        int m, int n, int p,
        int mn, int np, int mp
    ) {
        for (int r = start; r < end; r++) {
            int b = r / m;
            int i = r % m;
            int offsetA = b * mn;
            int offsetB = b * np;
            int offsetC = b * mp;
            int rowA = offsetA + i * n;
            int rowC = offsetC + i * p;

            for (int t = 0; t < n; t++) {
                float aVal = A[rowA + t];

                if (!transpose) {
                    int baseB = offsetB + t * p;
                    for (int j = 0; j < p; j++) {
                        C[rowC + j] += aVal * B[baseB + j];
                    }
                } else {
                    for (int j = 0; j < p; j++) {
                        C[rowC + j] += aVal * B[offsetB + j * n + t];
                    }
                }
            }
        }
    }

    private class ScalarAction extends RecursiveAction {
        private final int start, end;
        private final MatmulParameters parameters;

        private ScalarAction(int start, int end, MatmulParameters parameters) {
            this.start = start;
            this.end = end;
            this.parameters = parameters;
        }

        @Override
        protected void compute() {
            int work = end - start;

            if (isOverThreshold(work, parameters.np())) {
                int mid = (start + end) >>> 1;
                invokeAll(
                    new ScalarAction(start, mid, parameters),
                    new ScalarAction(mid, end, parameters)
                );
            } else {
                matmulBlock(
                    parameters.A(), parameters.B(), parameters.C(),
                    start, end,
                    parameters.transpose(),
                    parameters.m(), parameters.n(), parameters.p(),
                    parameters.mn(), parameters.np(), parameters.mp()
                );
            }
        }
    }
}
