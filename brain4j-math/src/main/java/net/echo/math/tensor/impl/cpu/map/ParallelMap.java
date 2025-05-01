package net.echo.math.tensor.impl.cpu.map;

import net.echo.math.lang.DoubleToDoubleFunction;

import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public class ParallelMap extends RecursiveAction {

    private static final int WORK_THRESHOLD = 1024;

    private final MapParameters parameters;
    private final int start;
    private final int end;

    public ParallelMap(MapParameters parameters, int start, int end) {
        this.parameters = parameters;
        this.start = start;
        this.end = end;
    }

    @Override
    protected void compute() {
        int work = end - start;
        if (work > WORK_THRESHOLD) {
            int mid = (start + end) >>> 1;
            invokeAll(
                    new ParallelMap(parameters, start, mid),
                    new ParallelMap(parameters, mid, end)
            );
            return;
        }

        DoubleToDoubleFunction function = parameters.function();
        float[] data = parameters.data();

        for (int i = start; i < end; i++) {
            double value = data[i];
            data[i] = (float) function.apply(value);
        }
    }

    public static void map(
            DoubleToDoubleFunction function, float[] data, ForkJoinPool pool
    ) {
        MapParameters parameters = new MapParameters(function, data);
        ParallelMap parallelMap = new ParallelMap(parameters, 0, data.length);
        pool.invoke(parallelMap);
    }

}