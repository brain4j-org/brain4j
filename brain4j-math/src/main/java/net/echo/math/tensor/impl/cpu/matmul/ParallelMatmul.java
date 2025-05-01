package net.echo.math.tensor.impl.cpu.matmul;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public class ParallelMatmul extends RecursiveAction {

    private static final int WORK_THRESHOLD = 1;
    private static final int COMPLEXITY_THRESHOLD = 65536;

    private final MatmulParameters parameters;
    private final int start;
    private final int end;

    public ParallelMatmul(MatmulParameters parameters, int start, int end) {
        this.parameters = parameters;
        this.start = start;
        this.end = end;
    }

    @Override
    protected void compute() {
        int n = parameters.n();
        int p = parameters.p();
        int np = parameters.np();

        int work = end - start;
        if (work > WORK_THRESHOLD && work * np > COMPLEXITY_THRESHOLD) {
            int mid = (start + end) >>> 1;
            invokeAll(
                    new ParallelMatmul(parameters, start, mid),
                    new ParallelMatmul(parameters, mid, end)
            );
            return;
        }

        int m = parameters.m();
        int mn = parameters.mn();
        int mp = parameters.mp();

        float[] A = parameters.A();
        float[] B = parameters.B();
        float[] C = parameters.C();

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
                int colB = offsetB + t * p;
                for (int j = 0; j < p; j++) {
                    C[rowC + j] += aVal * B[colB + j];
                }
            }
        }
    }

    public static void multiply(
            int batch, int m, int n, int p, float[] A, float[] B, float[] C, ForkJoinPool pool
    ) {
        MatmulParameters parameters = new MatmulParameters(m, n, p, A, B, C, n * p, m * n, m * p);
        ParallelMatmul parallelMatmul = new ParallelMatmul(parameters, 0, batch * m);
        pool.invoke(parallelMatmul);
    }

}