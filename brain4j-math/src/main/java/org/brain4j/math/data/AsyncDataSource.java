package org.brain4j.math.data;

import org.brain4j.math.Pair;
import org.brain4j.math.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

public class AsyncDataSource extends ListDataSource {

    public static final int PROCESSORS = Runtime.getRuntime().availableProcessors();
    public static final ExecutorService EXECUTOR = Executors.newFixedThreadPool(PROCESSORS);

    public AsyncDataSource(List<Sample> samples, boolean shuffle, int batchSize) {
        super(samples, shuffle, batchSize);
    }

    public void accept(Consumer<Pair<Tensor, Tensor>> task) {
        reset();

        List<CompletableFuture<Void>> futures = new ArrayList<>();

        while (hasNext()) {
            Pair<Tensor, Tensor> partition = nextBatch();
            futures.add(CompletableFuture.runAsync(() -> task.accept(partition), EXECUTOR));
        }

        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();
    }

    @Override
    public AsyncDataSource clone() {
        return new AsyncDataSource(samples, false, batchSize);
    }
}
