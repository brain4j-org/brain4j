package org.brain4j.core.model;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.training.BackPropagation;
import org.brain4j.core.training.EvaluationResult;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.training.updater.impl.StochasticUpdater;
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
 * Represents a feedforward neural network model.
 * <p>
 * Supports multiple layer types, loss functions, optimizers, and training via backpropagation.
 * Provides methods for training (fit), prediction, evaluation, and model summary.
 * </p>
 */
public class Model {

    protected final List<Layer> layers;

    protected BackPropagation backPropagation;
    protected Optimizer optimizer;
    protected Updater updater;
    protected LossFunction lossFunction;

    /**
     * Constructs a new model instance with the given layers.
     * @param layers the sequence of layers forming the neural network
     */
    public Model(Layer... layers) {
        this.layers = List.of(layers);
    }

    private void connectLayers() {
        Layer previous = layers.getFirst();
        Random random = Random.from(new SplittableRandom());

        for (int i = 1; i < size(); i++) {
            Layer layer = layerAt(i);
            Layer next = null;

            if (i < size() - 1) {
                next = layerAt(i + 1);
            }

            layer.connect(previous, next);

            int input = previous.size();
            int output = layer.size();

            layer.initWeights(random, input, output);

            previous = layer;
        }
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

    /**
     * Trains the model using the provided training dataset.
     * @param train the dataset for training
     */
    public void fit(ListDataSource train) {
        fit(train, train, 1, Integer.MAX_VALUE);
    }
    /**
     * Trains the model using the training dataset and evaluates on a validation dataset.
     * @param train the training dataset
     * @param validation the validation dataset for periodic evaluation
     */
    public void fit(ListDataSource train, ListDataSource validation) {
        fit(train, validation, 1, Integer.MAX_VALUE);
    }

    /**
     * Trains the model for a specified number of epochs.
     * @param train the training dataset
     * @param epoches number of epochs to train
     */
    public void fit(ListDataSource train, int epoches) {
        fit(train, train, epoches, Integer.MAX_VALUE);
    }

    /**
     * Trains the model for a specified number of epochs and evaluates on a validation dataset.
     * @param train the training dataset
     * @param validation the validation dataset for periodic evaluation
     * @param epoches number of epochs to train
     */
    public void fit(ListDataSource train, ListDataSource validation, int epoches) {
        fit(train, validation, epoches, Integer.MAX_VALUE);
    }

    /**
     * Trains the model for a specified number of epochs.
     * @param train the training dataset
     * @param epoches number of epochs to train
     * @param evaluateEvery frequency (in epochs) for evaluation
     */
    public void fit(ListDataSource train, int epoches, int evaluateEvery) {
        fit(train, train, epoches, evaluateEvery);
    }

    /**
     * Trains the model for a specified number of epochs and evaluates periodically.
     *
     * @param train the training dataset
     * @param validation the validation dataset for evaluation
     * @param epoches total epochs to train
     * @param evaluateEvery frequency (in epochs) for evaluation
     */
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

    /**
     * Performs a forward pass to predict output given an input tensor.
     *
     * @param input the input tensor to the network
     * @return the predicted output tensor
     * @throws IllegalArgumentException if input is null or has invalid dimensions
     */
    public Tensor predict(Tensor input) {
        return predict(new StatesCache(this), input);
    }

    /**
     * Performs a forward pass with cached states to predict output.
     *
     * @param cache the cache for storing intermediate states
     * @param input the input tensor
     * @return the output tensor predicted by the network
     */
    public Tensor predict(StatesCache cache, Tensor input) {
        return predict(cache, input, false);
    }

    /**
     * Performs a forward pass with optional training mode.
     *
     * @param cache the cache to store intermediate states
     * @param input the input tensor
     * @param training whether the network is in training mode (affects layers like dropout)
     * @return the predicted output tensor
     */
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

            if (layer == null || layer.skipForward()) {
                throw new IllegalStateException("Layer at index " + i + " is null!");
            }

            pass = layer.forward(cache, pass, i, training);
        }

        return pass;
    }

    /**
     * Evaluates the model on a given dataset, computing loss and classification metrics.
     *
     * @param dataSource the dataset to evaluate
     * @return an {@link EvaluationResult} containing loss, accuracy, and classification details
     */
    public EvaluationResult evaluate(ListDataSource dataSource) {
        int classes = Math.max(2, dataSource.samples().getFirst().label().elements());
        Map<Integer, Tensor> classifications = new ConcurrentHashMap<>();

        for (int i = 0; i < classes; i++) {
            classifications.put(i, Tensors.zeros(classes));
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

    /**
     * Computes the average loss of the model over the given dataset.
     *
     * @param dataSource the dataset to compute loss on
     * @return the average loss value
     */
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

    /**
     * Compiles the model by setting the loss function, optimizer, and default updater.
     *
     * @param lossFunction the loss function to use
     * @param optimizer the optimization algorithm
     * @return the compiled model instance for method chaining
     */
    public Model compile(LossFunction lossFunction, Optimizer optimizer) {
        return compile(lossFunction, optimizer, new StochasticUpdater());
    }

    /**
     * Compiles the model by setting the loss function, optimizer, and custom updater.
     *
     * @param lossFunction the loss function to use
     * @param optimizer the optimization algorithm
     * @param updater the updater managing gradient application
     * @return the compiled model instance for method chaining
     */
    public Model compile(LossFunction lossFunction, Optimizer optimizer, Updater updater) {
        this.optimizer = optimizer;
        this.updater = updater;
        this.lossFunction = lossFunction;
        this.backPropagation = new BackPropagation(this, optimizer, updater);

        this.updater.resetGradients(this);
        this.optimizer.initialize(this);

        connectLayers();
        return this;
    }

    /**
     * Returns a formatted summary of the model architecture,
     * including layers, neuron counts, weights, activations, and total parameters.
     *
     * @return a human-readable summary string of the model
     * @throws IllegalStateException if the model is not compiled before calling this method
     */
    public String summary() {
        if (updater == null || optimizer == null) {
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

            int neurons = layer.size();
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

    /**
     * Retrieves the layer at the specified index.
     *
     * @param index the index of the layer
     * @return the layer at the given index
     */
    public Layer layerAt(int index) {
        return layers.get(index);
    }

    /**
     * Returns the total number of layers in the model.
     * @return the number of layers
     */
    public int size() {
        return layers.size();
    }

    /**
     * Returns an immutable list of layers composing the model.
     * @return the list of layers
     */
    public List<Layer> layers() {
        return layers;
    }

    /**
     * Returns the optimizer currently used by the model.
     * @return the optimizer instance
     */
    public Optimizer optimizer() {
        return optimizer;
    }

    /**
     * Returns the updater currently used by the model.
     * @return the updater instance
     */
    public Updater updater() {
        return updater;
    }

    /**
     * Returns the loss function currently set in the model.
     * @return the loss function instance
     */
    public LossFunction lossFunction() {
        return lossFunction;
    }
}
