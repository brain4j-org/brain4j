package org.brain4j.core.training;

import org.brain4j.core.model.Model;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.Pair;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.device.DeviceType;
import org.brain4j.math.kernel.GpuContextHandler;
import org.brain4j.math.tensor.Tensor;

import java.util.function.BiConsumer;

public class BackPropagation {

    private final Model model;
    private final Optimizer optimizer;
    private final Updater updater;

    public BackPropagation(Model model, Optimizer optimizer, Updater updater) {
        this.model = model;
        this.optimizer = optimizer;
        this.updater = updater;
        updater.resetGradients(model);
    }

    public void propagatePartition(
        Pair<Tensor[], Tensor> partition,
        BiConsumer<Integer, Double> postBatchCallback,
        int index
    ) {
        Tensor[] inputs = partition.first();
        Tensor labels = partition.second();

        boolean isOnGpu = model.deviceType() == DeviceType.GPU;
        StatesCache cache = new StatesCache(isOnGpu);

        long start = System.nanoTime();

        Tensor output = model.predict(cache, true, inputs);
        model.backpropagate(cache, output, labels);

        int elements = 1;

        for (Tensor input : inputs) {
            elements *= input.shape()[0];
        }

        optimizer.postBatch();
        updater.postBatch(model, optimizer.learningRate(), elements);

        if (isOnGpu) {
            GpuContextHandler.closeQueue();
        }

        double took = (System.nanoTime() - start) / 1e6;
        postBatchCallback.accept(index, took);
    }

    public void iteration(ListDataSource dataSource, BiConsumer<Integer, Double> postBatchCallback) {
        dataSource.reset();

        while (dataSource.hasNext()) {
            Pair<Tensor[], Tensor> batch = dataSource.nextBatch();
            propagatePartition(batch, postBatchCallback, dataSource.cursor());
        }

        updater.postFit(model, optimizer.learningRate(), dataSource.size());
    }
}