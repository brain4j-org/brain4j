package net.echo.brain4j.model.impl;

import net.echo.brain4j.initialization.WeightInit;
import net.echo.brain4j.initialization.WeightInitializer;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.structure.StatesCache;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.optimizer.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.training.updater.impl.StochasticUpdater;
import net.echo.math.BrainUtils;
import net.echo.math.DataSet;
import net.echo.math.Pair;
import net.echo.math.data.ListDataSource;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.index.Range;

import java.util.List;
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
