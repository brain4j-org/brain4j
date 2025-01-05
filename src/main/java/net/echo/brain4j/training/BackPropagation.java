package net.echo.brain4j.training;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DropoutLayer;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.utils.GenericUtils;
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

        for (List<DataRow> partition : dataSet.getPartitions()) {
            List<Thread> threads = new ArrayList<>();

            for (DataRow row : partition) {
                Thread thread = Thread.startVirtualThread(() -> {
                    Vector output = model.predict(row.inputs());
                    Vector target = row.outputs();

                    backpropagate(target.toArray(), output.toArray());
                });

                threads.add(thread);
            }

            GenericUtils.waitAll(threads);

            updater.postBatch(model.getLayers(), optimizer.getLearningRate());
        }

        updater.postFit(model.getLayers(), optimizer.getLearningRate());
    }

    public void backpropagate(double[] targets, double[] outputs) {
        List<Layer> layers = model.getLayers();
        initialDelta(layers, targets, outputs);

        for (int l = layers.size() - 2; l > 0; l--) {
            Layer layer = layers.get(l);

            if (layer instanceof DropoutLayer dropoutLayer) {
                Layer previous = layers.get(l - 1);
                dropoutLayer.backward(previous.getNeurons());
                continue;
            }

            layer.propagate(updater, optimizer);
        }

        optimizer.postIteration(updater, layers);
        updater.postIteration(layers, optimizer.getLearningRate());
    }

    private void initialDelta(List<Layer> layers, double[] targets, double[] outputs) {
        Layer outputLayer = layers.getLast();

        for (int i = 0; i < outputLayer.getNeurons().size(); i++) {
            Neuron neuron = outputLayer.getNeuronAt(i);

            double output = outputs[i];
            double error = targets[i] - output;

            double delta = error * outputLayer.getActivation().getFunction().getDerivative(output);
            neuron.setDelta(delta);
        }
    }
}