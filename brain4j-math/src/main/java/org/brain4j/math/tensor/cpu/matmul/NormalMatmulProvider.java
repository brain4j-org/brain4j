package org.brain4j.math.tensor.cpu.matmul;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public class NormalMatmulProvider implements MatmulProvider {

    private static final int WORK_THRESHOLD = 1;
    private static final int COMPLEXITY_THRESHOLD = 65536;

    private static class ScalarAction extends RecursiveAction {

        private final MatmulParameters parameters;
        private final int batchA;
        private final int batchB;
        private final int start;
        private final int end;

        public ScalarAction(MatmulParameters parameters, int batchA, int batchB, int start, int end) {
            this.parameters = parameters;
            this.batchA = batchA;
            this.batchB = batchB;
            this.start = start;
            this.end = end;
        }

        @Override
        protected void compute() {
            int n = parameters.n();
            int p = parameters.p();
            int np = parameters.np();

            int work = end - start;
            if (isOverThreshold(work, np)) {
                int mid = (start + end) >>> 1;
                invokeAll(
                    new ScalarAction(parameters, batchA, batchB, start, mid),
                    new ScalarAction(parameters, batchA, batchB, mid, end)
                );
                return;
            }

            int m = parameters.m();
            int mn = parameters.mn();
            int mp = parameters.mp();

            float[] A = parameters.A();
            float[] B = parameters.B();
            float[] C = parameters.C();

            multiplySection(start, end, m, n, p, A, B, C, mn, np, mp, batchA, batchB);
        }
    }

    public void multiply(
        ForkJoinPool pool,
        int batch,
        int m, int n, int p,
        float[] A, float[] B, float[] C,
        int batchA, int batchB
    ) {
        int start = 0;
        int end = batch * m;
        int mn = m * n;
        int np = n * p;
        int mp = m * p;

        int work = end - start;

        if (!isOverThreshold(work, np)) {
            multiplySection(start, end, m, n, p, A, B, C, mn, np, mp, batchA, batchB);
            return;
        }

        MatmulParameters parameters = new MatmulParameters(m, n, p, A, B, C, np, mn, mp, batchA, batchB);
        ScalarAction action = new ScalarAction(parameters, batchA, batchB, start, end);
        pool.invoke(action);
    }

    private static void multiplySection(
        int start, int end,
        int m, int n, int p,
        float[] A, float[] B, float[] C,
        int mn, int np, int mp,
        int batchA, int batchB
    ) {
        for (int r = start; r < end; r++) {
            int batchIndex = r / m;
            int i = r % m;

            int batchIndexA = batchA == 1 ? 0 : batchIndex;
            int batchIndexB = batchB == 1 ? 0 : batchIndex;

            int offsetA = batchIndexA * mn;
            int offsetB = batchIndexB * np;
            int offsetC = batchIndex * mp;

            int rowA = offsetA + i * n;
            int rowC = offsetC + i * p;

            for (int t = 0; t < n; t++) {
                float aVal = A[rowA + t];
                int colB = offsetB + t * p;

                for (int j = 0; j < p; j++) {
                    C[rowC + j] += aVal * B[colB + j];
                }
            }
        }
    }

    private static boolean isOverThreshold(int work, int np) {
        return work > WORK_THRESHOLD && work * np > COMPLEXITY_THRESHOLD;
    }
}