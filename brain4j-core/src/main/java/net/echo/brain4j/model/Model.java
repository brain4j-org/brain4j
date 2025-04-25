package net.echo.brain4j.model;

import net.echo.brain4j.Brain4J;
import net.echo.brain4j.adapters.Adapter;
import net.echo.brain4j.adapters.ModernAdapter;
import net.echo.brain4j.initialization.WeightInit;
import net.echo.brain4j.initialization.WeightInitializer;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DropoutLayer;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.model.impl.Transformer;
import net.echo.brain4j.structure.StatesCache;
import net.echo.brain4j.training.BackPropagation;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.evaluation.EvaluationResult;
import net.echo.brain4j.training.optimizer.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.training.updater.impl.StochasticUpdater;
import net.echo.math.BrainUtils;
import net.echo.math.DataSet;
import net.echo.math.Pair;
import net.echo.math.data.ListDataSource;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.Tensors;
import net.echo.math.tensor.index.Range;
import net.echo.math.vector.Vector;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Represents an abstract neural network model.
 * Inputs and outputs are represented by n-dimensional tensors.
 *
 * @see Sequential Sequential
 * @see Transformer Transformer
 */
public abstract class Model implements Adapter {

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
        Layer previousLayer = layers.getFirst();

        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            layer.compile(weightInit, lossFunction, optimizer, updater);

            if (i > 0 && layer.canPropagate()) {
                int inputNeurons = previousLayer.getTotalNeurons();
                int outputNeurons = layer.getTotalNeurons();

                double bound = weightInit.getBound(inputNeurons, outputNeurons);

                layer.connect(generator, previousLayer, null, bound);
                previousLayer = layer;
            }
        }
    }

    public abstract Thread makeEvaluation(Pair<Tensor, Tensor> batch, Map<Integer, Tensor> classifications, AtomicReference<Double> totalLoss);

    public abstract void fit(ListDataSource dataSource);

    public abstract Tensor predict(StatesCache cache, Tensor input, boolean training);

    @Deprecated
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
        for (int i = 0; i < epoches; i++) {
            propagation.iteration(dataSource);

            int currentEpoch = i + 1;

            if (Brain4J.isLogging()) {
                printProgressBar(currentEpoch, epoches, evaluateEvery);
            }

            if (currentEpoch % evaluateEvery == 0) {
                EvaluationResult result = evaluate(dataSource.clone());
                System.out.printf("Loss at epoch %s: %.4f | Accuracy: %.2f%%\n", currentEpoch, result.loss(), result.accuracy() * 100);
            }
        }
    }

    public void printProgressBar(int currentEpoch, int epoches, int evaluateEvery) {
        int progressBarLength = 30;
        double percentage = (double) currentEpoch / epoches;

        int repetitions = (int) (percentage * progressBarLength);
        int remaining = progressBarLength - repetitions;

        String progressBar = "\u001B[32m" + Brain4J.getHeaderChar().repeat(repetitions) + "\u001B[0m" + "━".repeat(remaining);
        System.out.printf("\rEpoch: %s/%s %-30s %.2f%%", currentEpoch, epoches, progressBar, percentage * 100);

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

        String sizeOfParams = BrainUtils.formatNumber(params * 4); // 4 = float size in bytes
        String sizeOfWeights = BrainUtils.formatNumber(totalWeights * 4);
        String sizeOfBiases = BrainUtils.formatNumber(totalBiases * 4);

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
