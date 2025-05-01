package org.brain4j.math.tensor.impl.cpu.map;

import org.brain4j.math.lang.DoubleToDoubleFunction;

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
        if (isOverThreshold(work)) {
            int mid = (start + end) >>> 1;
            invokeAll(
                    new ParallelMap(parameters, start, mid),
                    new ParallelMap(parameters, mid, end)
            );
            return;
        }

        DoubleToDoubleFunction function = parameters.function();
        float[] data = parameters.data();

        mapSection(start, end, data, function);
    }

    public static void map(
            DoubleToDoubleFunction function, float[] data, ForkJoinPool pool
    ) {
        int start = 0;
        int end = data.length;

        int work = end - start;
        if (!isOverThreshold(work)) {
            mapSection(start, end, data, function);
            return;
        }

        MapParameters parameters = new MapParameters(function, data);
        ParallelMap parallelMap = new ParallelMap(parameters, 0, data.length);
        pool.invoke(parallelMap);
    }

    private static void mapSection(int start, int end, float[] data, DoubleToDoubleFunction function) {
        for (int i = start; i < end; i++) {
            double value = data[i];
            data[i] = (float) function.apply(value);
        }
    }

    private static boolean isOverThreshold(int work) {
        return work > WORK_THRESHOLD;
    }

}