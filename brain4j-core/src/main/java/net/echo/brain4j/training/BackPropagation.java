package net.echo.brain4j.training;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.StatesCache;
import net.echo.brain4j.training.optimizer.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.math.Pair;
import net.echo.math.data.ListDataSource;
import net.echo.math.tensor.Tensor;

import java.util.List;

public class BackPropagation {

    private final Model model;
    private final Optimizer optimizer;
    private final Updater updater;

    public BackPropagation(Model model, Optimizer optimizer, Updater updater) {
        this.model = model;
        this.optimizer = optimizer;
        this.updater = updater;
    }

    public void propagatePartition(Pair<Tensor, Tensor> partition) {
        Tensor inputs = partition.first();
        Tensor labels = partition.second();

        StatesCache cache = new StatesCache();
        Tensor output = model.predict(cache, inputs, true);

        backpropagation(cache, labels, output);

        int elements = inputs.shape()[0];

        optimizer.postBatch();
        updater.postBatch(model, optimizer.getLearningRate(), elements);
    }

    public void iteration(ListDataSource dataSource) {
        dataSource.reset();

        while (dataSource.hasNext()) {
            propagatePartition(dataSource.nextBatch());
        }

        updater.postFit(model, optimizer.getLearningRate(), dataSource.size());
    }

    public void backpropagation(StatesCache cache, Tensor targets, Tensor outputs) {
        List<Layer> layers = model.getLayers();
        LossFunction lossFunction = model.getLossFunction();

        Layer last = layers.getLast();
        Tensor delta = last.computeLoss(cache, targets, outputs, lossFunction);

        for (int l = layers.size() - 2; l >= 1; l--) {
            Layer layer = layers.get(l);

            if (!layer.canPropagate()) continue;

            delta = layer.backward(cache, last, delta);
            last = layer;
        }

        optimizer.postIteration(cache, updater, layers);
    }
}