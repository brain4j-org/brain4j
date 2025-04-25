package net.echo.math.data;

import net.echo.math.Pair;
import net.echo.math.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

public class AsyncDataSource extends ListDataSource {

    public AsyncDataSource(List<Sample> samples, boolean shuffle, int batches) {
        super(samples, shuffle, batches);
    }

    public void propagate(Consumer<Pair<Tensor, Tensor>> task) {
        int processors = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(processors);

        reset();

        while (hasNext()) {
            Pair<Tensor, Tensor> partition = nextBatch();
            executor.submit(() -> task.accept(partition));
        }

        executor.close();
    }

    @Override
    public AsyncDataSource clone() {
        return new AsyncDataSource(samples, false, batches);
    }
}
