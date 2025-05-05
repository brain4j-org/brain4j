package org.brain4j.core.model;

import org.brain4j.core.Brain4J;
import org.brain4j.core.adapters.BinarySerializable;
import org.brain4j.core.adapters.ModernAdapter;
import org.brain4j.core.initialization.WeightInit;
import org.brain4j.core.initialization.WeightInitializer;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.DropoutLayer;
import org.brain4j.core.loss.Loss;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.model.impl.Sequential;
import org.brain4j.core.model.impl.Transformer;
import org.brain4j.core.structure.StatesCache;
import org.brain4j.core.training.BackPropagation;
import org.brain4j.core.training.evaluation.EvaluationResult;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.training.updater.impl.StochasticUpdater;
import org.brain4j.math.BrainUtils;
import org.brain4j.math.Pair;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;
import org.brain4j.math.tensor.index.Range;
import org.brain4j.math.vector.Vector;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;

import static org.brain4j.math.constants.Constants.*;

/**
 * Represents an abstract neural network model.
 * Inputs and outputs are represented by n-dimensional tensors.
 *
 * @see Sequential Sequential
 * @see Transformer Transformer
 */
public abstract class Model implements BinarySerializable {

    protected BackPropagation propagation;
    protected List<Layer> layers;
    protected WeightInitializer weightInit;
    protected LossFunction lossFunction;
    protected Optimizer optimizer;
    protected Updater updater;
    protected Random generator;
    protected int seed;

    public Model(Layer... layers) {
        this.layers = new ArrayList<>(Arrays.asList(layers));
    }

    public void connect(WeightInitializer weightInit) {
        Layer previousLayer = null;

        for (Layer layer : layers) {
            layer.compile(weightInit, lossFunction, optimizer, updater);

            if (layer.canPropagate()) {
                double bound = 0;

                if (previousLayer != null) {
                    int inputNeurons = previousLayer.getTotalNeurons();
                    int outputNeurons = layer.getTotalNeurons();

                    bound = weightInit.getBound(inputNeurons, outputNeurons);
                }

                layer.connect(generator, previousLayer, bound);
                previousLayer = layer;
            }
        }
    }

    public abstract Thread makeEvaluation(Pair<Tensor, Tensor> batch, Map<Integer, Tensor> classifications, AtomicReference<Double> totalLoss);

    public abstract void fit(ListDataSource dataSource);

    public abstract Tensor predict(StatesCache cache, Tensor input, boolean training);

    @Deprecated(since = "2.8.0", forRemoval = true)
    public Vector predict(Vector input) {
        Tensor tensor = Tensors.vector(input.toArray());
        Tensor output = predict(new StatesCache(), tensor, false);

        return Vector.of(output.getData());
    }

    public EvaluationResult evaluate(ListDataSource dataSource) {
        int classes = dataSource.getSamples().getFirst().label().elements();

        // Binary classification
        if (classes == 1) {
            classes = 2;
        }

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

        BrainUtils.waitAll(threads);

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

        BrainUtils.waitAll(threads);
        return totalError.get() / dataSource.size();
    }

    public void fit(ListDataSource dataSource, int epoches) {
        fit(dataSource, epoches, Integer.MAX_VALUE);
    }

    public void fit(ListDataSource dataSource, int epoches, int evaluateEvery) {
        fit(dataSource, dataSource, epoches, evaluateEvery);
    }

    public void fit(ListDataSource trainSource, ListDataSource testSource, int epoches) {
        fit(trainSource, testSource, epoches, Integer.MAX_VALUE);
    }

    public void fit(ListDataSource trainSource, ListDataSource testSource, int epoches, int evaluateEvery) {
        for (int i = 0; i < epoches; i++) {
            long start = System.nanoTime();
            propagation.iteration(trainSource);
            double tookMs = (System.nanoTime() - start) / 1e6;

            int currentEpoch = i + 1;

            if (Brain4J.isLogging()) {
                printProgressBar(tookMs, currentEpoch, epoches, evaluateEvery);
            }

            if (currentEpoch % evaluateEvery == 0) {
                EvaluationResult result = evaluate(testSource.clone());
                System.out.printf("[%s/%s] Loss: %.4f | Accuracy: %.2f%% | F1-Score: %.2f%%\n", currentEpoch, epoches,
                        result.loss(), result.accuracy() * 100, result.f1Score() * 100);
            }
        }
    }


    public void printProgressBar(double tookMs, int currentEpoch, int epoches, int evaluateEvery) {
        int progressBarLength = 20;
        double percentage = (double) currentEpoch / epoches;

        int repetitions = (int) (percentage * progressBarLength);
        int remaining = progressBarLength - repetitions;

        String barChar = Brain4J.getHeaderChar();
        int remainingEpoches = epoches - currentEpoch;

        double seconds = tookMs / 1000;
        double remainingTime = seconds * remainingEpoches;

        String remainingTimeStr = BrainUtils.formatDuration(remainingTime);
        String timeStr = BrainUtils.formatDuration(seconds);

        String heading = WHITE + "[%s/%s] ";
        String progressBar = BLUE + barChar.repeat(repetitions) + RESET + barChar.repeat(remaining);
        String percentual = YELLOW + " %.2f%%" + RESET;
        String time = GRAY + " [%s/epoch | %s remaining]" + RESET;

        String message = String.format(heading + progressBar + percentual + time,
                currentEpoch, epoches, percentage * 100, timeStr, remainingTimeStr);
        System.out.print("\r" + message);

        if (currentEpoch == epoches || currentEpoch % evaluateEvery == 0) {
            System.out.println();
        }
    }

    public Tensor predict(StatesCache cache, Tensor input) {
        return predict(cache, input, false);
    }

    public Tensor predict(Tensor input) {
        return predict(new StatesCache(), input, false);
    }

    public Model compile(Loss function, Optimizer optimizer) {
        return compile(function.getFunction(), optimizer);
    }

    public Model compile(Loss function, Optimizer optimizer, Updater updater) {
        return compile(WeightInit.UNIFORM_XAVIER.getFunction(), function.getFunction(), optimizer, updater);
    }

    public Model compile(LossFunction function, Optimizer optimizer) {
        return compile(WeightInit.UNIFORM_XAVIER.getFunction(), function, optimizer, new StochasticUpdater());
    }

    public Model compile(WeightInitializer initializer, LossFunction lossFunction, Optimizer optimizer, Updater updater) {
        this.propagation = new BackPropagation(this, optimizer, updater);
        this.weightInit = initializer;
        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
        this.updater = updater;
        this.generator = Random.from(new SplittableRandom(seed));

        return this;
    }

    public Model compile(WeightInit initializer, Loss lossFunction, Optimizer optimizer, Updater updater) {
        return compile(initializer.getFunction(), lossFunction.getFunction(), optimizer, updater);
    }

    public void load(String path) throws Exception {
        ModernAdapter.deserialize(path, this);
    }

    public void save(String path) throws Exception {
        ModernAdapter.serialize(path, this);
    }

    public Model add(Layer layer) {
        layers.add(layer);
        return this;
    }

    public Model add(Layer... layers) {
        this.layers.addAll(Arrays.asList(layers));
        return this;
    }

    public String summary() {
        StringBuilder stats = new StringBuilder();
        DecimalFormat format = new DecimalFormat("#,###");

        String pattern = "%-7s %-20s %-12s %-15s %-15s\n";
        String divider = BrainUtils.getHeader(" Architecture ", Brain4J.getHeaderChar());

        stats.append(divider);
        stats.append(pattern.formatted("Index", "Layer", "Neurons", "Weights", "Activation")).append("\n");

        long totalWeights = 0;
        long totalBiases = 0;

        for (int i = 0; i < this.layers.size(); i++) {
            Layer layer = this.layers.get(i);

            String layerType = layer.getClass().getSimpleName();

            int neurons = layer.getTotalNeurons();
            int weights = layer.getTotalParams();

            String formatNeurons = layer instanceof DropoutLayer ? "-" : format.format(neurons);
            String formatWeights = format.format(weights);

            stats.append(pattern.formatted(i, layerType, formatNeurons, formatWeights, layer.getActivation().getName()));

            totalWeights += weights;
            totalBiases += neurons;
        }

        long params = totalWeights + totalBiases;

        String parameters = format.format(params);
        String weights = format.format(totalWeights);
        String biases = format.format(totalBiases);

        byte floatSize = Float.BYTES; // 4 bytes
        String sizeOfParams = BrainUtils.formatNumber(params * floatSize);
        String sizeOfWeights = BrainUtils.formatNumber(totalWeights * floatSize);
        String sizeOfBiases = BrainUtils.formatNumber(totalBiases * floatSize);

        stats.append(BrainUtils.getHeader(" Recap ", Brain4J.getHeaderChar()));
        stats.append("Total weights: %s (%s)\n".formatted(weights, sizeOfWeights));
        stats.append("Total biases: %s (%s)\n".formatted(biases, sizeOfBiases));
        stats.append("Total parameters: %s (%s)\n".formatted(parameters, sizeOfParams));
        stats.append(BrainUtils.getHeader("", Brain4J.getHeaderChar()));

        return stats.toString();
    }

    public void setPropagation(BackPropagation propagation) {
        this.propagation = propagation;
    }

    public BackPropagation getPropagation() {
        return propagation;
    }

    public LossFunction getLossFunction() {
        return lossFunction;
    }

    public void setLossFunction(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }

    public Optimizer getOptimizer() {
        return optimizer;
    }

    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
    }

    public Updater getUpdater() {
        return updater;
    }

    public void setUpdater(Updater updater) {
        this.updater = updater;
    }

    public WeightInitializer getWeightInit() {
        return weightInit;
    }

    public void setWeightInit(WeightInitializer weightInit) {
        this.weightInit = weightInit;
    }

    public Random getGenerator() {
        return generator;
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public void setSeed(int seed) {
        this.seed = seed;
        this.generator = Random.from(new SplittableRandom(seed));
    }

    public int getSeed() {
        return seed;
    }

    public int getTotalNeurons() {
        int total = 0;

        for (Layer layer : layers) {
            total += layer.getTotalNeurons();
        }

        return total;
    }

    public int getTotalWeights() {
        int total = 0;

        for (Layer layer : layers) {
            total += layer.getTotalParams();
        }

        return total;
    }

    protected Thread predictPartition(Pair<Tensor, Tensor> partition, AtomicReference<Double> totalError) {
        return Thread.startVirtualThread(() -> {
            Tensor inputs = partition.first();
            Tensor targets = partition.second();
            Tensor outputs = predict(inputs);

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

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        stream.writeInt(layers.size());

        for (Layer layer : layers) {
            stream.writeUTF(layer.getClass().getName());
            layer.serialize(stream);
        }
    }

    @Override
    public void deserialize(DataInputStream stream) throws Exception {
        int layersSize = stream.readInt();
        layers.clear();

        for (int i = 0; i < layersSize; i++) {
            String layerClassPath = stream.readUTF();
            Layer instance = BrainUtils.newInstance(layerClassPath);

            instance.deserialize(stream);
            layers.add(instance);
        }
    }
}
