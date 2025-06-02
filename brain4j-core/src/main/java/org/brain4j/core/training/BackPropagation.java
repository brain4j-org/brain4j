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

    public void propagatePartition(Pair<Tensor, Tensor> partition) {
        long start = System.nanoTime();
        Tensor inputs = partition.first();
        Tensor labels = partition.second();

        StatesCache cache = new StatesCache(model);
        Tensor output = model.predict(cache, inputs, true);

        backpropagation(cache, labels, output);

        int elements = inputs.shape()[0];

        optimizer.postBatch();
        updater.postBatch(model, optimizer.learningRate(), elements);
        long took = System.nanoTime() - start;
        System.out.println("Took " + (took / 1e6) + " ms to compute batch");
    }

    public void iteration(ListDataSource dataSource) {
        dataSource.accept(this::propagatePartition);
        updater.postFit(model, optimizer.learningRate(), dataSource.size());
    }

    public void backpropagation(StatesCache cache, Tensor targets, Tensor outputs) {
        List<Layer> layers = model.layers();
        LossFunction lossFunction = model.lossFunction();

        int count = layers.size() - 1;

        Layer last = layers.getLast();
        last.computeLoss(updater, cache, targets, outputs, lossFunction, count);

        for (int l = count - 1; l >= 0; l--) {
            Layer layer = layers.get(l);

            if (layer.skipPropagate()) continue;

            layer.backward(updater, optimizer, l);
        }
    }
}