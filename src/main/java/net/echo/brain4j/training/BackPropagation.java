package net.echo.brain4j.training;

import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.threading.NeuronCacheHolder;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.utils.MLUtils;
import net.echo.brain4j.utils.Vector;

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

    public void iterate(DataSet dataSet) {
        if (!dataSet.isPartitioned()) {
            throw new RuntimeException("Dataset must be partitioned, use DataSet#partition(batches) before training.");
        }

        List<Thread> threads = new ArrayList<>();

        for (List<DataRow> partition : dataSet.getPartitions()) {
            Thread thread = Thread.startVirtualThread(() -> {
                NeuronCacheHolder batchHolder = new NeuronCacheHolder();

                for (DataRow row : partition) {
                    NeuronCacheHolder cacheHolder = new NeuronCacheHolder();

                    Vector output = model.predict(cacheHolder, row.inputs());
                    Vector target = row.outputs();

                    backpropagate(cacheHolder, target.toArray(), output.toArray());
                }

                updater.postBatch(batchHolder, model, optimizer.getLearningRate());
            });

            threads.add(thread);
        }

        MLUtils.waitAll(threads);
        updater.postFit(model, optimizer.getLearningRate());
    }

    public void backpropagate(NeuronCacheHolder cacheHolder, double[] targets, double[] outputs) {
        List<Layer> layers = model.getLayers();
        initialDelta(cacheHolder, layers, targets, outputs);

        Layer previous = null;

        for (int l = layers.size() - 2; l > 0; l--) {
            Layer layer = layers.get(l);

            if (!layer.canPropagate()) continue;

            layer.propagate(cacheHolder, previous, updater);
            previous = layer;
        }

        optimizer.postIteration(cacheHolder, updater, layers);
        updater.postIteration(model, optimizer.getLearningRate());
    }

    private void initialDelta(NeuronCacheHolder cacheHolder, List<Layer> layers, double[] targets, double[] outputs) {
        Layer outputLayer = layers.getLast();

        List<Neuron> neurons = outputLayer.getNeurons();
        Vector result = new Vector(outputs.length);

        for (int i = 0; i < outputs.length; i++) {
            result.set(i, outputs[i]);
        }

        Activation function = outputLayer.getActivation().getFunction();
        Vector changes = function.getDerivative(result);

        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);

            double output = outputs[i];
            double error = targets[i] - output;

            double delta = changes.get(i) * error;
            neuron.setDelta(cacheHolder, delta);
        }
    }
}