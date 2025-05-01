package org.brain4j.core.model.impl;

import org.brain4j.core.initialization.WeightInit;
import org.brain4j.core.initialization.WeightInitializer;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.loss.Loss;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.model.Model;
import org.brain4j.core.structure.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.training.updater.impl.StochasticUpdater;
import org.brain4j.math.BrainUtils;
import org.brain4j.math.Pair;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;

import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

public class Transformer extends Model {

    public Transformer(Layer... layers) {
        super(layers);
    }

    @Override
    public Thread makeEvaluation(Pair<Tensor, Tensor> batch, Map<Integer, Tensor> classifications, AtomicReference<Double> totalLoss) {
        return Thread.startVirtualThread(() -> {
            Tensor inputs = batch.first();
            Tensor outputs = batch.second();

            int batchSize = inputs.shape()[0];

            for (int i = 0; i < batchSize; i++) {
                Range range = new Range(i, i + 1);

                Tensor input = inputs.slice(range).vector();
                Tensor output = outputs.slice(range).vector();

                Tensor prediction = predict(input);

                double loss = lossFunction.calculate(output, prediction);
                totalLoss.updateAndGet(v -> v + loss);

                int predIndex = BrainUtils.argmax(prediction);
                int targetIndex = BrainUtils.argmax(output);

                Tensor predictions = classifications.get(targetIndex);
                int pred = (int) predictions.get(predIndex);

                predictions.set(pred + 1, predIndex);
            }
        });
    }

    @Override
    public Transformer compile(LossFunction function, Optimizer optimizer) {
        return compile(WeightInit.UNIFORM_XAVIER.getFunction(), function, optimizer, new StochasticUpdater());
    }

    @Override
    public Transformer compile(Loss function, Optimizer optimizer) {
        return compile(function.getFunction(), optimizer);
    }

    @Override
    public Transformer compile(WeightInit initializer, Loss lossFunction, Optimizer optimizer, Updater updater) {
        return compile(initializer.getFunction(), lossFunction.getFunction(), optimizer, updater);
    }

    @Override
    public Transformer compile(WeightInitializer initializer, LossFunction lossFunction, Optimizer optimizer, Updater updater) {
        super.compile(initializer, lossFunction, optimizer, updater);

        connect(initializer);

        return this;
    }

    @Override
    public void fit(ListDataSource dataSource) {
        propagation.iteration(dataSource);
    }

    @Override
    public Tensor predict(StatesCache cache, Tensor input, boolean training) {
        if (!cache.isCompatibleWithCache(input)) {
            cache.markAsNewSession();
        }

        Tensor result = input;

        for (Layer layer : layers) {
            result = layer.forward(cache, layer, result, training);
        }

        return result;
    }
}
