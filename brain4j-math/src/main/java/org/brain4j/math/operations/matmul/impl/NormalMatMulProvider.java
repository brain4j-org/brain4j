package org.brain4j.math.operations.matmul.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.operations.matmul.MatmulParameters;
import org.brain4j.math.operations.matmul.MatmulProvider;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class NormalMatMulProvider implements MatmulProvider {
    
    private static final int PARALLELISM = Runtime.getRuntime().availableProcessors();
    private static final int WORK_THRESHOLD = 8;
    private static final int COMPLEXITY_THRESHOLD = 65536 * 4;

    private static boolean isOverThreshold(int work, int np) {
        return work > WORK_THRESHOLD && work * np > COMPLEXITY_THRESHOLD;
    }

    @Override
    public void multiply(Tensor a, Tensor b, Tensor c) {
        float[] A = a.data();
        float[] B = b.data();
        float[] C = c.data();

        int rankA = a.rank();
        int rankB = b.rank();

        int[] shapeA = a.shape();
        int[] shapeB = b.shape();
        int[] shapeC = c.shape();

        int batchA = 1;
        int batchB = 1;
        int batch = 1;

        for (int i = 0; i < rankA - 2; i++) batchA *= shapeA[i];
        for (int i = 0; i < rankB - 2; i++) batchB *= shapeB[i];
        for (int i = 0; i < shapeC.length - 2; i++) batch *= shapeC[i];

        int m = shapeA[rankA - 2];
        int n = shapeA[rankA - 1];
        int p = shapeB[rankB - 1];

        MatmulParameters parameters = new MatmulParameters(A, B, C, m, n, p, a.transposed(), b.transposed(), batchA, batchB);

        int work = batch * m;

        int mn = m * n;
        int np = n * p;
        int mp = m * p;

        if (!isOverThreshold(work, np)) {
            matmulBlock(A, B, C, 0, work, m, n, p, mn, np, mp, batchA, batchB, a.transposed(), b.transposed());
            return;
        }


        int step = work / PARALLELISM;
        List<ScalarAction> actions = new ArrayList<>();

        for (int i = 0; i < PARALLELISM; i++) {
            int startIndex = i * step;
            int endIndex = (i == PARALLELISM - 1) ? work : Math.min(startIndex + step, work);

            if (startIndex < endIndex) {
                actions.add(new ScalarAction(parameters, startIndex, endIndex));
            }
        }

        ForkJoinTask.invokeAll(actions);
    }
    
    private void matmulBlock(
        float[] a, float[] b, float[] c,
        int start, int end,
        int m, int n, int p,
        int mn, int np, int mp,
        int batchA, int batchB,
        boolean transposedA, boolean transposedB
    ) {
        if (batchA == 1 && batchB == 1) {
            matmulSimple(a, b, c, start, end, m, n, p, mn, np, mp, transposedA, transposedB);
        } else {
            matmulGeneric(a, b, c, start, end, m, n, p, mn, np, mp, batchA, batchB, transposedA, transposedB);
        }
    }
    
    private void matmulSimple(
        float[] a, float[] b, float[] c,
        int start, int end,
        int m, int n, int p,
        int mn, int np, int mp,
        boolean transposedA, boolean transposedB
    ) {
        int offsetRowA = transposedA ? 1 : n;
        int offsetAccessA = transposedA ? m : 1;
        
        int offsetRowB = transposedB ? 1 : p;
        int offsetAccessB = transposedB ? n : 1;
        
        for (int r = start; r < end; r++) {
            int batch = r / m;
            int i = r % m;
            int offsetA = batch * mn;
            int offsetB = batch * np;
            int offsetC = batch * mp;
            int rowA = offsetA + i * offsetRowA;
            int rowC = offsetC + i * p;
            
            for (int t = 0; t < n; t++) {
                float aVal = a[rowA + t * offsetAccessA];
                int colB = offsetB + t * offsetRowB;
                
                for (int j = 0; j < p; j++) {
                    c[rowC + j] += aVal * b[colB + j * offsetAccessB];
                }
            }
        }
    }
    
    private void matmulGeneric(
        float[] a, float[] b, float[] c,
        int start, int end,
        int m, int n, int p,
        int mn, int np, int mp,
        int batchA, int batchB,
        boolean transposedA, boolean transposedB
    ) {
        int offsetRowA = transposedA ? 1 : n;
        int offsetAccessA = transposedA ? m : 1;
        
        int offsetRowB = transposedB ? 1 : p;
        int offsetAccessB = transposedB ? n : 1;
        
        for (int r = start; r < end; r++) {
            int offsetA = (batchA == 1 ? 0 : r / m) * mn;
            int offsetB = (batchB == 1 ? 0 : r / m) * np;
            int ci = (r / m) * mp;
            int i = r % m;
            int rowA = offsetA + i * offsetRowA;
            int rowC = ci + i * p;

            for (int t = 0; t < n; t++) {
                float aVal = a[rowA + t * offsetAccessA];
                int colB = offsetB + t * offsetRowB;

                for (int j = 0; j < p; j++) {
                    c[rowC + j] += aVal * b[colB + j * offsetAccessB];
                }
            }
        }
    }
    
    private class ScalarAction extends RecursiveAction {

        private final MatmulParameters parameters;
        private final int start, end;

        private ScalarAction(MatmulParameters parameters, int start, int end) {
            this.parameters = parameters;
            this.start = start;
            this.end = end;
        }

        @Override
        protected void compute() {
            int np = parameters.np();
            int work = end - start;

            if (isOverThreshold(work, np)) {
                int mid = (start + end) >>> 1;
                invokeAll(
                    new ScalarAction(parameters, start, mid),
                    new ScalarAction(parameters, mid, end)
                );
            } else {
                matmulBlock(
                    parameters.A(), parameters.B(), parameters.C(),
                    start, end,
                    parameters.m(), parameters.n(), parameters.p(),
                    parameters.mn(), parameters.np(), parameters.mp(),
                    parameters.batchA(), parameters.batchB(),
                    parameters.transposedA(), parameters.transposedB()
                );
            }
        }
    }
}
