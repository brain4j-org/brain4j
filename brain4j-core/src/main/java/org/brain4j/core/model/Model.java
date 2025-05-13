package org.brain4j.core.model;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.loss.Loss;
import org.brain4j.core.training.BackPropagation;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.training.updater.impl.StochasticUpdater;
import org.brain4j.core.weights.WeightInit;
import org.brain4j.math.data.AsyncDataSource;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;

import java.util.List;
import java.util.Random;
import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicReference;

public class Model {

    private final List<Layer> layers;

    private BackPropagation backPropagation;
    private Optimizer optimizer;
    private Updater updater;
    private WeightInit weightInit;
    private Loss lossFunction;

    public Model(Layer... layers) {
        this.layers = List.of(layers);
    }

    public void fit(ListDataSource train) {
        fit(train, train);
    }

    public void fit(ListDataSource train, ListDataSource validation) {
        fit(train, validation, 1);
    }

    public void fit(ListDataSource train, ListDataSource validation, int epoches) {
        fit(train, validation, epoches, Integer.MAX_VALUE);
    }

    public void fit(ListDataSource train, ListDataSource validation, int epoches, int evaluateEvery) {
        for (int i = 1; i <= epoches; i++) {
            backPropagation.iteration(train);
        }
    }

    public Tensor predict(Tensor input) {
        return predict(new StatesCache(this), input);
    }

    public Tensor predict(StatesCache cache, Tensor input) {
        return predict(cache, input, false);
    }

    public Tensor predict(StatesCache cache, Tensor input, boolean training) {
        if (input == null || input.dimension() == 0) {
            throw new IllegalArgumentException("Input is either null or has dimension of 0!");
        }

        if (input.dimension() < 2) {
            // Shape: [batch_size, input_size]
            input = input.reshape(1, input.elements());
        }

        int[] shape = input.shape();

        Layer inputLayer = layers.getFirst();

        if (inputLayer.size() != shape[1]) {
            throw new IllegalArgumentException(
                "Input shape does not match. Expected: " + inputLayer.size() + ", Received: " + shape[1]
            );
        }

        Tensor pass = input.withGrad();

        cache.setInput(0, input);
        cache.setOutput(0, pass);

        for (int i = 1; i < size(); i++) {
            Layer layer = layerAt(i);

            if (layer == null) {
                throw new IllegalStateException("Layer at index " + i + " is null!");
            }

            if (!layer.canPropagate()) continue;

            pass = layer.forward(cache, pass, i, training);
        }

        return pass;
    }

    public double loss(ListDataSource source) {
        AsyncDataSource asyncSource = new AsyncDataSource(source.samples(), false, source.batchSize());
        AtomicReference<Double> loss = new AtomicReference<>(0.0);

        asyncSource.accept(batch -> {
            Tensor input = batch.first();
            Tensor label = batch.second();

            Tensor prediction = predict(input);

            int batchSize = input.shape()[0];

            for (int i = 0; i < batchSize; i++) {
                Range range = new Range(i, i + 1);

                Tensor samplePrediction = prediction.slice(range).vector();
                Tensor sampleLabel = label.slice(range).vector();

                double error = lossFunction.function().calculate(sampleLabel, samplePrediction);
                loss.updateAndGet(v -> v + error);
            }
        });

        return loss.get();
    }

    public Model compile(Optimizer optimizer, Loss loss) {
        return compile(optimizer, new StochasticUpdater(), WeightInit.UNIFORM_XAVIER, loss);
    }

    public Model compile(Optimizer optimizer, Updater updater, WeightInit weightInit, Loss loss) {
        this.optimizer = optimizer;
        this.updater = updater;
        this.weightInit = weightInit;
        this.lossFunction = loss;
        this.backPropagation = new BackPropagation(this, optimizer, updater);
        connectLayers();
        return this;
    }

    private void connectLayers() {
        Layer previous = layers.getFirst();
        Random random = Random.from(new SplittableRandom());

        for (int i = 1; i < size(); i++) {
            Layer layer = layerAt(i);

            int input = previous.size();
            int output = layer.size();

            double bound = weightInit.function().getBound(input, output);

            layer.connect(previous);
            layer.initWeights(random, bound);

            previous = layer;
        }
    }

    public Layer layerAt(int index) {
        return layers.get(index);
    }

    public int size() {
        return layers.size();
    }

    public List<Layer> layers() {
        return layers;
    }

    public Optimizer optimizer() {
        return optimizer;
    }

    public Updater updater() {
        return updater;
    }

    public WeightInit weightInit() {
        return weightInit;
    }

    public Loss lossFunction() {
        return lossFunction;
    }
}
