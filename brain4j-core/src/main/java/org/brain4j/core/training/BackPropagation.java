package org.brain4j.core.training;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.model.Model;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.math.Pair;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.tensor.Tensor;

import java.util.List;
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
        Pair<Tensor, Tensor> partition,
        BiConsumer<Integer, Double> postBatchCallback,
        int index
    ) {
        Tensor inputs = partition.first();
        Tensor labels = partition.second();

        StatesCache cache = new StatesCache(model);

        long start = System.nanoTime();

        Tensor output = model.predict(cache, true, inputs);
        backpropagation(cache, labels, output);

        int elements = inputs.shape()[0];

        optimizer.postBatch();
        updater.postBatch(model, optimizer.learningRate(), elements);

        double took = (System.nanoTime() - start) / 1e6;
        postBatchCallback.accept(index, took);

    }

    public void iteration(ListDataSource dataSource, BiConsumer<Integer, Double> postBatchCallback) {
        dataSource.reset();

        while (dataSource.hasNext()) {
            Pair<Tensor, Tensor> batch = dataSource.nextBatch();
            propagatePartition(batch, postBatchCallback, dataSource.cursor());
        }

        updater.postFit(model, optimizer.learningRate(), dataSource.size());
    }

    public void backpropagation(StatesCache cache, Tensor targets, Tensor outputs) {
        List<Layer> flattened = model.flattened();
        LossFunction lossFunction = model.lossFunction();

        int count = flattened.size() - 1;

        Layer last = flattened.getLast();
        last.computeLoss(updater, cache, targets, outputs, lossFunction, count);

        for (int l = count; l >= 0; l--) {
            Layer layer = flattened.get(l);

            if (layer.skipPropagate()) continue;

            layer.backward(updater, optimizer, l);
        }
    }
}