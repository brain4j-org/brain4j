package net.echo.brain4j.training;

import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.utils.DataSet;
import net.echo.brain4j.utils.Vector;
import org.checkerframework.checker.units.qual.A;

import java.util.ArrayList;
import java.util.List;

import static net.echo.brain4j.utils.MLUtils.waitAll;

public class BackPropagation {

    private final Sequential model;
    private final Optimizer optimizer;
    private final Updater updater;

    public BackPropagation(Sequential model, Optimizer optimizer, Updater updater) {
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

                Vector output = model.predict(cacheHolder, row.inputs());
                Vector target = row.outputs();

                backpropagate(cacheHolder, target, output);
            });

            threads.add(thread);
        }

        waitAll(threads);
        updater.postBatch(model, optimizer.getLearningRate());
    }

    public void iteration(DataSet<DataRow> dataSet) {
        partitionIfRequired(dataSet);

        for (List<DataRow> partition : dataSet.getPartitions()) {
            propagatePartition(partition);
        }

        updater.postFit(model, optimizer.getLearningRate());
    }

    public void backpropagate(StatesCache cacheHolder, Vector targets, Vector outputs) {
        List<Layer<?, ?>> layers = model.getLayers();
        initializeDeltas(cacheHolder, layers, targets, outputs);

        Layer<?, ?> previous = null;

        for (int l = layers.size() - 2; l > 0; l--) {
            Layer<?, ?> layer = layers.get(l);

            if (!layer.canPropagate()) continue;

            layer.propagate(cacheHolder, previous, updater, optimizer);
            previous = layer;
        }

        optimizer.postIteration(cacheHolder, updater, layers);
        updater.postIteration(model, optimizer.getLearningRate());
    }

    private void initializeDeltas(StatesCache cacheHolder, List<Layer<?, ?>> layers, Vector targets, Vector outputs) {
        Layer<?, ?> outputLayer = layers.getLast();

        List<Neuron> neurons = outputLayer.getNeurons();
        Activation function = outputLayer.getActivation().getFunction();

        Vector derivative = function.getDerivative(outputs);

        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);

            double error = outputs.get(i) - targets.get(i);
            double delta = model.getLossFunction().getDelta(error, derivative.get(i));

            neuron.setDelta(cacheHolder, (float) delta);
        }
    }
}