package org.brain4j.core.model;

import org.brain4j.core.Brain4J;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.training.BackPropagation;
import org.brain4j.core.training.EvaluationResult;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.training.updater.impl.StochasticUpdater;
import org.brain4j.core.weights.WeightInitialization;
import org.brain4j.core.weights.impl.UniformXavierInit;
import org.brain4j.math.Commons;
import org.brain4j.math.Pair;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;
import org.brain4j.math.tensor.index.Range;

import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;

import static org.brain4j.math.constants.Constants.*;
import static org.brain4j.math.constants.Constants.GRAY;
import static org.brain4j.math.constants.Constants.LIGHT_GREEN;
import static org.brain4j.math.constants.Constants.LIGHT_YELLOW;
import static org.brain4j.math.constants.Constants.RESET;
import static org.brain4j.math.constants.Constants.WHITE;

/**
 * Implementation of a simple neural network. Supports multiple types of layers, loss functions, and optimizers.
 */
public class Model {

    protected final List<Layer> layers;

    protected BackPropagation backPropagation;
    protected Optimizer optimizer;
    protected Updater updater;
    protected WeightInitialization weightInit;
    protected LossFunction lossFunction;

    /**
     * Constructs a new model instance with the given layers.
     * @param layers the layers of this neural network
     */
    public Model(Layer... layers) {
        this.layers = List.of(layers);
    }
    
    protected Thread makeEvaluation(Pair<Tensor, Tensor> batch, Map<Integer, Tensor> classifications, AtomicReference<Double> totalLoss) {
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

    protected Thread predictPartition(Pair<Tensor, Tensor> partition, AtomicReference<Double> totalError) {
        return Thread.startVirtualThread(() -> {
            Tensor inputs = partition.first();
            Tensor targets = partition.second();
            Tensor outputs = predict(new StatesCache(this), inputs, true);

            int batchSize = outputs.shape()[0];

            for (int i = 0; i < batchSize; i++) {
                Range range = new Range(i, i + 1);

                Tensor output = outputs.slice(range).vector();
                Tensor target = targets.slice(range).vector();

                double loss = lossFunction.calculate(target, output);
                totalError.updateAndGet(v -> v + loss);
            }
        });
    }

    protected void printEvaluation(int step, int epoches, ListDataSource testSource) {
        EvaluationResult result = evaluate(testSource.clone());

        String lossMsg = "Loss: " + MAGENTA + "%.4f" + RESET;
        String accuracyMsg = "Accuracy: " + LIGHT_BLUE + "%.2f%%" + RESET;
        String f1ScoreMsg = "F1-Score: " + LIGHT_GREEN + "%.2f%%" + RESET;

        String message = "[%s/%s] " + lossMsg + " | " + accuracyMsg + " | " + f1ScoreMsg + "\n";
        System.out.printf(message, step, epoches, result.loss(), result.accuracy() * 100, result.f1Score() * 100);
    }

    protected void printProgressBar(
        double tookMs,
        int currentEpoch,
        int epoches,
        int evaluateEvery
    ) {
        int progressBarLength = 20;
        double percentage = (double) currentEpoch / epoches;

        String barChar = Commons.getHeaderChar();
        int remainingEpoches = epoches - currentEpoch;

        double seconds = tookMs / 1000;
        double remainingTime = seconds * remainingEpoches;

        String remainingTimeStr = Commons.formatDuration(remainingTime);
        String timeStr = Commons.formatDuration(seconds);

        String progressMsg = WHITE + "[%s/%s] ";
        String progressBar = LIGHT_GREEN + Commons.createProgressBar(
            percentage,
            progressBarLength,
            barChar,
            RESET + barChar
        );

        String percentual = LIGHT_YELLOW + " %.2f%%" + RESET;
        String time = GRAY + " [%s/epoch | %s remaining]" + RESET;
        String message = String.format(progressMsg + progressBar + percentual + time,
            currentEpoch, epoches, percentage * 100, timeStr, remainingTimeStr);

        System.out.print("\r" + message);

        if (currentEpoch == epoches || currentEpoch % evaluateEvery == 0) {
            System.out.println();
        }
    }

    public void fit(ListDataSource train) {
        fit(train, train, 1, Integer.MAX_VALUE);
    }

    public void fit(ListDataSource train, ListDataSource validation) {
        fit(train, validation, 1, Integer.MAX_VALUE);
    }

    public void fit(ListDataSource train, ListDataSource validation, int epoches) {
        fit(train, validation, epoches, Integer.MAX_VALUE);
    }

    public void fit(ListDataSource train, ListDataSource validation, int epoches, int evaluateEvery) {
        for (int i = 1; i <= epoches; i++) {
            long start = System.nanoTime();
            backPropagation.iteration(train);
            long tookNanos = System.nanoTime() - start;

            printProgressBar(tookNanos / 1e6, i, epoches, evaluateEvery);

            if (i % evaluateEvery == 0) {
                printEvaluation(i, epoches, validation);
            }
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

        Tensor pass = input;

        cache.setInput(0, input);
        cache.setOutput(0, pass);

        for (int i = 0; i < size(); i++) {
            Layer layer = layerAt(i);

            if (layer == null) {
                throw new IllegalStateException("Layer at index " + i + " is null!");
            }

            if (layer.skipPropagate()) continue;

            pass = layer.forward(cache, pass, i, training);
        }

        return pass;
    }

    public EvaluationResult evaluate(ListDataSource dataSource) {
        int classes = Math.max(2, dataSource.samples().getFirst().label().elements());
        Map<Integer, Tensor> classifications = new ConcurrentHashMap<>();

        for (int i = 0; i < classes; i++) {
            classifications.put(i, Tensors.create(classes));
        }

        List<Thread> threads = new ArrayList<>();
        AtomicReference<Double> totalLoss = new AtomicReference<>(0.0);

        dataSource.reset();

        while (dataSource.hasNext()) {
            Pair<Tensor, Tensor> partition = dataSource.nextBatch();
            threads.add(makeEvaluation(partition, classifications, totalLoss));
        }

        Commons.waitAll(threads);

        return new EvaluationResult(totalLoss.get() / dataSource.size(), classes, classifications);
    }

    public double loss(ListDataSource dataSource) {
        AtomicReference<Double> totalError = new AtomicReference<>(0.0);
        List<Thread> threads = new ArrayList<>();

        dataSource.reset();

        while (dataSource.hasNext()) {
            Pair<Tensor, Tensor> partition = dataSource.nextBatch();
            threads.add(predictPartition(partition, totalError));
        }

        Commons.waitAll(threads);

        return totalError.get() / dataSource.size();
    }

    public Model compile(LossFunction lossFunction, Optimizer optimizer) {
        return compile(lossFunction, optimizer, new StochasticUpdater(), new UniformXavierInit());
    }

    public Model compile(LossFunction lossFunction, Optimizer optimizer, Updater updater, WeightInitialization weightInit) {
        this.optimizer = optimizer;
        this.updater = updater;
        this.weightInit = weightInit;
        this.lossFunction = lossFunction;
        this.backPropagation = new BackPropagation(this, optimizer, updater);

        this.updater.resetGradients(this);
        this.optimizer.initialize(this);

        connectLayers();
        return this;
    }


    public String summary() {
        if (updater == null || optimizer == null || weightInit == null) {
            throw new IllegalStateException("The network is not compiled! Make sure to call compile() before.");
        }

        StringBuilder stats = new StringBuilder();
        DecimalFormat format = new DecimalFormat("#,###");

        String pattern = "%-7s %-20s %-12s %-15s %-15s\n";
        String divider = Commons.getHeader(" Architecture ", Commons.getHeaderChar());

        stats.append(divider);
        stats.append(pattern.formatted("Index", "Layer", "Neurons", "Weights", "Activation")).append("\n");

        long totalWeights = 0;
        long totalBiases = 0;

        for (int i = 0; i < this.layers.size(); i++) {
            Layer layer = this.layers.get(i);

            String layerType = layer.getClass().getSimpleName();

            int neurons = layer.totalNeurons();
            int weights = layer.totalWeights();

            String formatNeurons = neurons == 0 ? "-" : format.format(neurons);
            String formatWeights = weights == 0 ? "-" : format.format(weights);

            stats.append(pattern.formatted(i, layerType, formatNeurons, formatWeights, layer.activation().getName()));

            totalWeights += weights;
            totalBiases += neurons;
        }

        long params = totalWeights + totalBiases;

        String parameters = format.format(params);
        String weights = format.format(totalWeights);
        String biases = format.format(totalBiases);

        byte floatSize = Float.BYTES; // 4 bytes
        String sizeOfParams = Commons.formatNumber(params * floatSize);
        String sizeOfWeights = Commons.formatNumber(totalWeights * floatSize);
        String sizeOfBiases = Commons.formatNumber(totalBiases * floatSize);

        stats.append(Commons.getHeader(" Recap ", Commons.getHeaderChar()));
        stats.append("Total weights: %s (%s)\n".formatted(weights, sizeOfWeights));
        stats.append("Total biases: %s (%s)\n".formatted(biases, sizeOfBiases));
        stats.append("Total parameters: %s (%s)\n".formatted(parameters, sizeOfParams));
        stats.append(Commons.getHeader("", Commons.getHeaderChar()));

        return stats.toString();
    }

    private void connectLayers() {
        Layer previous = layers.getFirst();
        Random random = Random.from(new SplittableRandom());

        for (int i = 1; i < size(); i++) {
            Layer layer = layerAt(i);
            layer.connect(previous);

            int input = previous.size();
            int output = layer.size();

            double bound = weightInit.getBound(input, output);

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

    public WeightInitialization weightInit() {
        return weightInit;
    }

    public LossFunction lossFunction() {
        return lossFunction;
    }
}
