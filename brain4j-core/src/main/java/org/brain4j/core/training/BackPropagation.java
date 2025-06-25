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
        Pair<Tensor[], Tensor> batch,
        BiConsumer<Integer, Double> postBatchCallback,
        int index
    ) {
        DeviceType device = model.deviceType();
        StatesCache cache = new StatesCache(device);

        long start = System.nanoTime();

        Tensor[] inputs = batch.first();
        Tensor labels = batch.second();

        Tensor output = model.predict(cache, true, inputs);
        model.backpropagate(cache, output, labels);

        int elements = 1;

        for (Tensor input : inputs) {
            elements *= input.shape()[0]; // first element is ALWAYS the batch size
        }

        optimizer.postBatch();
        updater.postBatch(model, optimizer.learningRate(), elements);

        if (device == DeviceType.GPU) {
            GpuContextHandler.closeQueue();
        }

        double took = (System.nanoTime() - start) / 1e6;
        postBatchCallback.accept(index, took);
    }

    public void iteration(ListDataSource dataSource, BiConsumer<Integer, Double> postBatchCallback) {
        dataSource.reset();

        while (dataSource.hasNext()) {
            Pair<Tensor[], Tensor> batch = hostTo(dataSource.nextBatch());
            propagatePartition(batch, postBatchCallback, dataSource.cursor());
        }

        updater.postFit(model, optimizer.learningRate(), dataSource.size());
    }

    private Pair<Tensor[], Tensor> hostTo(Pair<Tensor[], Tensor> partition) {
        DeviceType device = model.deviceType();

        Tensor[] inputs = partition.first();
        Tensor labels = partition.second().to(device);

        for (int i = 0; i < inputs.length; i++) {
            Tensor input = inputs[i];
            inputs[i] = input.to(device);
        }

        return new Pair<>(inputs, labels);
    }
}