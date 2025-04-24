package net.echo.brain4j.model.impl;

import net.echo.brain4j.initialization.WeightInit;
import net.echo.brain4j.initialization.WeightInitializer;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.LayerNorm;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.StatesCache;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.optimizer.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.training.updater.impl.StochasticUpdater;
import net.echo.math.DataSet;
import net.echo.math.data.ListDataSource;
import net.echo.math.tensor.Tensor;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Implementation of a sequential neural network model.
 * This model processes an input {@link Tensor} and produces an label {@link Tensor}.
 * It supports training using instances of {@link DataRow}.
 * </p>
 */
public class Sequential extends Model {

    public Sequential(Layer... layers) {
        super(layers);
    }

    @Override
    public Thread makeEvaluation(List<DataRow> partition, Map<Integer, Tensor> classifications, AtomicReference<Double> totalLoss) {
        return Thread.startVirtualThread(() -> {
            for (DataRow row : partition) {
                Tensor prediction = predict(row.inputs());
                Tensor expected = row.outputs();

                int predIndex = prediction.argmax();
                int targetIndex = expected.argmax();

                if (row.outputs().elements() == 1) {
                    predIndex = prediction.get(0) > 0.5 ? 1 : 0;
                    targetIndex = (int) row.outputs().get(0);
                }

                double loss = lossFunction.calculate(expected, prediction);
                totalLoss.updateAndGet(v -> v + loss);

                Tensor predictions = classifications.get(targetIndex);
                int pred = (int) predictions.get(predIndex);

                predictions.set(pred + 1, predIndex);
            }
        });
    }

    @Override
    public Sequential compile(Loss function, Optimizer optimizer) {
        return compile(function.getFunction(), optimizer);
    }

    @Override
    public Sequential compile(LossFunction function, Optimizer optimizer) {
        return compile(WeightInit.UNIFORM_XAVIER.getFunction(), function, optimizer, new StochasticUpdater());
    }

    @Override
    public Sequential compile(WeightInitializer initializer, LossFunction lossFunction, Optimizer optimizer, Updater updater) {
        super.compile(initializer, lossFunction, optimizer, updater);

        connect(initializer);

        this.optimizer.postInitialize(this);
        this.updater.postInitialize();

        return this;
    }

    @Override
    public Sequential compile(WeightInit initializer, Loss lossFunction, Optimizer optimizer, Updater updater) {
        return compile(initializer.getFunction(), lossFunction.getFunction(), optimizer, updater);
    }

    @Override
    public void fit(ListDataSource dataSet) {
        propagation.iteration(dataSet);
    }

    @Override
    public Tensor predict(Tensor input) {
        return predict(new StatesCache(), input, false);
    }

    @Override
    public Tensor predict(StatesCache cache, Tensor input, boolean training) {
        Layer workingLayer = layers.getFirst();

        if (input.dimension() == 1) {
            input = input.reshape(1, input.elements());
        }

        int elements = input.shape()[1];

        if (elements != workingLayer.getTotalNeurons()) {
            throw new IllegalArgumentException("Input dimensions do not match! (Input != Expected): %s != %s"
                    .formatted(elements, workingLayer.getTotalNeurons()));
        }

        Tensor denseResult = input.clone();

        cache.setInputTensor(workingLayer, denseResult);
        cache.setOutputTensor(workingLayer, denseResult);

        for (int l = 1; l < layers.size(); l++) {
            Layer layer = layers.get(l);

            cache.setInputTensor(layer, denseResult);

            denseResult = layer.forward(cache, workingLayer, denseResult, training);

            if (layer.canPropagate() && !(layer instanceof LayerNorm)) {
                workingLayer = layer;
            }
        }

        return denseResult;
    }
}
