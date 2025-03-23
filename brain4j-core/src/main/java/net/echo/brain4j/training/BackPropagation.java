package net.echo.brain4j.training;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.math4j.BrainUtils;
import net.echo.math4j.DataSet;
import net.echo.math4j.math.tensor.Tensor;

import java.util.ArrayList;
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

    private void partitionIfRequired(DataSet<DataRow> dataSet) {
        if (dataSet.isPartitioned()) return;

        int threads = Runtime.getRuntime().availableProcessors();
        int partitions = Math.min(threads, dataSet.getData().size());

        dataSet.partition(partitions);
    }

    private void propagatePartition(List<DataRow> partition) {
        List<Thread> threads = new ArrayList<>();

        for (DataRow row : partition) {
            Thread thread = Thread.startVirtualThread(() -> {
                StatesCache cacheHolder = new StatesCache();

                Tensor output = model.predict(cacheHolder, row.inputs(), true);
                Tensor target = row.outputs();

                backpropagation(cacheHolder, target, output);
            });

            threads.add(thread);
        }

        BrainUtils.waitAll(threads);
        // updater.postBatch(model, optimizer.getLearningRate());
    }

    public void iteration(DataSet<DataRow> dataSet) {
        partitionIfRequired(dataSet);

        for (List<DataRow> partition : dataSet.getPartitions()) {
            propagatePartition(partition);
        }

        updater.postFit(model, optimizer.getLearningRate());
    }

    public void backpropagation(StatesCache cacheHolder, Tensor targets, Tensor outputs) {
        List<Layer<?, ?>> layers = model.getLayers();
        initializeDeltas(cacheHolder, layers, targets, outputs);

        Layer<?, ?> previous = null;

        for (int l = layers.size() - 2; l > 0; l--) {
            Layer<?, ?> layer = layers.get(l);

            if (!layer.canPropagate()) continue;

            layer.propagate(cacheHolder, previous);
            previous = layer;
        }

        optimizer.postIteration(cacheHolder, updater, layers);
        updater.postIteration(model, optimizer.getLearningRate());
    }

    private void initializeDeltas(StatesCache cache, List<Layer<?, ?>> layers, Tensor targets, Tensor outputs) {
        Layer<?, ?> outputLayer = layers.getLast();
        LossFunction lossFunction = model.getLossFunction();

        outputLayer.computeLoss(cache, targets, outputs, lossFunction);
    }
}