package org.brain4j.core.model.impl;

import org.brain4j.core.initialization.WeightInit;
import org.brain4j.core.initialization.WeightInitializer;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.LayerNorm;
import org.brain4j.core.loss.Loss;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.model.Model;
import org.brain4j.core.structure.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.training.updater.impl.StochasticUpdater;
import org.brain4j.math.Pair;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;

import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

/**
 * A sequential neural network model implementation.
 * This model takes an input {@link Tensor} and outputs a corresponding label {@link Tensor}.
 * It supports training with data provided through instances of {@link ListDataSource}.
 */
public class Sequential extends Model {

    /**
     * Constructs a sequential model.
     * @param layers The layers of this model.
     */
    public Sequential(Layer... layers) {
        super(layers);
    }

    @Override
    public Thread makeEvaluation(Pair<Tensor, Tensor> batch, Map<Integer, Tensor> classifications, AtomicReference<Double> totalLoss) {
        return Thread.startVirtualThread(() -> {
            Tensor inputs = batch.first(); // [batch_size, input_size]
            Tensor expected = batch.second(); // [batch_size, output_size]

            Tensor prediction = predict(inputs); // [batch_size, output_size]
            int batchSize = inputs.shape()[0];

            for (int i = 0; i < batchSize; i++) {
                Range range = new Range(i, i + 1);

                Tensor output = prediction.slice(range).vector();
                Tensor target = expected.slice(range).vector();

                int predIndex = output.argmax();
                int targetIndex = target.argmax();

                if (output.elements() == 1) {
                    predIndex = output.get(0) > 0.5 ? 1 : 0;
                    targetIndex = (int) target.get(0);
                }

                double loss = lossFunction.calculate(target, output);
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
        this.updater.resetGradients();

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

            denseResult = layer.forward(cache, denseResult, training);

            if (layer.canPropagate() && !(layer instanceof LayerNorm)) {
                workingLayer = layer;
            }
        }

        return denseResult;
    }
}
