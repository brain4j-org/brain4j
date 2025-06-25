package org.brain4j.math.tensor.impl.cpu.matmul;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class SimdMatmulProvider implements MatmulProvider {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private static final int PARALLELISM = Runtime.getRuntime().availableProcessors();
    private static final int PARALLEL_COMPLEXITY_THRESHOLD = 65536;
    private static final int PARALLEL_WORK_THRESHOLD = PARALLELISM;

    private static final int SPLIT_COMPLEXITY_THRESHOLD = 65536 * 4;
    private static final int SPLIT_WORK_THRESHOLD = 2;

    private static class VectorAction extends RecursiveAction {

        private final MatmulParameters parameters;
        private final int start;
        private final int end;

        public VectorAction(MatmulParameters parameters, int start, int end) {
            this.parameters = parameters;
            this.start = start;
            this.end = end;
        }

        @Override
        protected void compute() {
            int n = parameters.n();
            int p = parameters.p();
            int np = parameters.np();

            int m = parameters.m();
            int mn = parameters.mn();
            int mp = parameters.mp();

            float[] A = parameters.A();
            float[] B = parameters.B();
            float[] C = parameters.C();

            int batchA = parameters.batchA();
            int batchB = parameters.batchB();

            int work = end - start;

            if (!isOverSplitThreshold(work, np)) {
                multiplySection(start, end, m, n, p, A, B, C, mn, np, mp, batchA, batchB);
                return;
            }

            int mid = (start + end) >>> 1;
            invokeAll(
                    new VectorAction(parameters, start, mid),
                    new VectorAction(parameters, mid, end)
            );
        }

    }

    @Override
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
        if (!isOverParallelThreshold(work, np)) {
            multiplySection(start, end, m, n, p, A, B, C, mn, np, mp, batchA, batchB);
            return;
        }

        int parallelism = PARALLELISM;
        int step = work / parallelism;

        MatmulParameters parameters = new MatmulParameters(m, n, p, A, B, C, np, mn, mp, batchA, batchB);
        int chunkSize = Math.max(32, (end - start + parallelism - 1) / parallelism);
        List<RecursiveAction> actions = new ArrayList<>();

        for (int i = start; i < end; i += chunkSize) {
            int to = Math.min(i + chunkSize, end);
            actions.add(new VectorAction(parameters, i, to));
        }
        ForkJoinTask.invokeAll(actions);
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

                int j = 0;

                for (; j < SPECIES.loopBound(p); j += SPECIES.length()) {
                    FloatVector vb = FloatVector.fromArray(SPECIES, B, colB + j);
                    FloatVector vc = FloatVector.fromArray(SPECIES, C, rowC + j);
                    vc.add(vb.mul(aVal)).intoArray(C, rowC + j);
                }

                for (; j < p; j++) {
                    C[rowC + j] += aVal * B[colB + j];
                }
            }
        }
    }

    private static boolean isOverParallelThreshold(int work, int np) {
        return work > PARALLEL_WORK_THRESHOLD && work * np > PARALLEL_COMPLEXITY_THRESHOLD;
    }

    private static boolean isOverSplitThreshold(int work, int np) {
        return work > SPLIT_WORK_THRESHOLD && work * np > SPLIT_COMPLEXITY_THRESHOLD;
    }
}