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

    public AsyncDataSource(List<Sample> samples, int batches) {
        super(samples, batches);
    }

    public void propagate(Consumer<Pair<Tensor, Tensor>> task) {
        List<Callable<Void>> tasks = new ArrayList<>();

        int processors = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(processors);

        reset();
        while (hasNext()) {
            Pair<Tensor, Tensor> partition = nextBatch();
            tasks.add(() -> {
                task.accept(partition);
                return null;
            });
        }

        try {
            executor.invokeAll(tasks);
            executor.close();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public AsyncDataSource clone() {
        return new AsyncDataSource(samples, batches);
    }
}
