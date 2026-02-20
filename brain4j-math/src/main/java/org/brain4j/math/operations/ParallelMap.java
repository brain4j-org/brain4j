package org.brain4j.math.operations;

import org.brain4j.math.commons.D2DFunction;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

public class ParallelMap extends RecursiveAction {
    
    public record MapParameters(D2DFunction function, float[] data) { }
    
    private static final int PARALLELISM = Runtime.getRuntime().availableProcessors();
    private static final int PARALLEL_WORK_THRESHOLD = PARALLELISM;
    private static final int PARALLEL_COMPLEXITY_THRESHOLD = 1 << 12; // 4096
    private static final int SPLIT_COMPLEXITY_THRESHOLD = 1 << 10; // 1024
    
    private static boolean isOverParallelThreshold(int work) {
        return work > PARALLEL_WORK_THRESHOLD && work > PARALLEL_COMPLEXITY_THRESHOLD;
    }
    
    private static boolean isOverSplitThreshold(int work) {
        return work > SPLIT_COMPLEXITY_THRESHOLD;
    }

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

        if (isOverSplitThreshold(work)) {
            int mid = (start + end) >>> 1;
            invokeAll(
                new ParallelMap(parameters, start, mid),
                new ParallelMap(parameters, mid, end)
            );
            return;
        }

        D2DFunction function = parameters.function();
        float[] data = parameters.data();

        mapSection(function, start, end, data);
    }

    public static void map(D2DFunction function, float[] data) {
        int start = 0;
        int end = data.length;
        int work = end - start;

        if (!isOverParallelThreshold(work)) {
            mapSection(function, start, end, data);
            return;
        }

        int step = work / PARALLELISM;
        var params = new MapParameters(function, data);
        List<ParallelMap> actions = new ArrayList<>();

        for (int i = 0; i < PARALLELISM; i++) {
            int startIndex = i * step;
            int endIndex = (i == PARALLELISM - 1) ? work : Math.min(startIndex + step, work);

            if (startIndex < endIndex) {
                actions.add(new ParallelMap(params, startIndex, endIndex));
            }
        }

        ForkJoinTask.invokeAll(actions);
    }

    private static void mapSection(
        D2DFunction function,
        int start,
        int end,
        float[] data
    ) {
        for (int i = start; i < end; i++) {
            data[i] = (float) function.apply(data[i]);
        }
    }
}