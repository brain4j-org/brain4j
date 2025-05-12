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

                double loss = lossFunction.calculate(target, output);
                totalLoss.updateAndGet(v -> v + loss);

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

        this.optimizer.postInitialize(this);
        this.updater.resetGradients();

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

        for (int i = 0; i < layers.size(); i++) {
            result = layers.get(i).forward(i, cache, result, training);
        }

        return result;
    }
}
