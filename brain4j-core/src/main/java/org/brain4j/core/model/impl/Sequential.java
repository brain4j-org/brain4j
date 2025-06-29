package org.brain4j.core.model.impl;

import org.brain4j.common.activation.Activation;
import org.brain4j.common.device.Device;
import org.brain4j.common.device.DeviceUtils;
import org.brain4j.common.tensor.impl.cpu.CpuTensor;
import org.brain4j.common.tensor.impl.gpu.GpuTensor;
import org.brain4j.core.Brain4J;
import org.brain4j.core.activation.Activations;
import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.layer.impl.DenseLayer;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.model.Model;
import org.brain4j.core.training.BackPropagation;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.training.updater.impl.StochasticUpdater;
import org.brain4j.core.training.wrappers.EvaluationResult;
import org.brain4j.common.Commons;
import org.brain4j.common.Pair;
import org.brain4j.common.data.ListDataSource;
import org.brain4j.common.kernel.GpuContextHandler;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.Tensors;
import org.brain4j.common.tensor.index.Range;
import org.jocl.cl_program;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.stream.IntStream;

import static org.brain4j.common.constants.Constants.*;

/**
 * Represents a simple feedforward neural network model.
 * <p>
 * Supports multiple layer types, loss functions, optimizers, and training via backpropagation.
 * Provides methods for training (fit), prediction, evaluation, and model summary.
 * </p>
 *
 * @author xEcho1337
 * @since 3.0
 * @see Transformer
 * @see AutoEncoder
 * @see MultiModel
 */
public class Sequential extends Layer implements Model {

    private static final Logger logger = LoggerFactory.getLogger(Sequential.class);
    private static final Logger training = LoggerFactory.getLogger("training");

    /* Data structures in the model */
    protected final List<Layer> layers;
    protected final List<Layer> flattened;

    /* The device the model is hosted on */
    protected Device device;

    /* General training parameters */
    protected BackPropagation backPropagation;
    protected Optimizer optimizer;
    protected Updater updater;
    protected LossFunction lossFunction;
    protected long seed;

    /**
     * Constructs a new neural network with the given layers.
     * @param layers the sequence of layers forming the neural network
     */
    public static Sequential of(Layer... layers) {
        return new Sequential(layers);
    }

    protected Sequential(Layer... layers) {
        this.layers = new ArrayList<>(List.of(layers));
        this.flattened = new ArrayList<>();
        this.seed = System.currentTimeMillis();

        for (Layer layer : layers) {
            if (layer instanceof Model subModel) {
                flattened.addAll(subModel.flattened());
                continue;
            }

            flattened.add(layer);
        }
    }

    private void connectLayers() {
        if (layers.isEmpty()) return;

        Layer previous = null;
        int size = flattened.size();

        for (int i = 0; i < size; i++) {
            Layer layer = flattenedAt(i);
            previous = layer.connect(previous);
        }

        int[] inputSizes = new int[size];

        for (int i = 1; i < inputSizes.length; i++) {
            inputSizes[i] = flattenedAt(i - 1).size();
        }

        IntStream.range(1, size).parallel().forEach(i -> {
            Layer layer = flattenedAt(i);

            int input = inputSizes[i];
            int output = layer.size();

            Random localRandom = Random.from(new SplittableRandom(seed + i));
            layer.initWeights(localRandom, input, output);
        });
    }

    protected void makeEvaluation(
        Pair<Tensor[], Tensor> batch,
        Map<Integer, Tensor> classifications,
        AtomicReference<Double> totalLoss
    ) {
        Tensor[] inputs = batch.first(); // [batch_size, input_size]
        Tensor expected = batch.second(); // [batch_size, output_size]

        Tensor prediction = predict(inputs).cpu(); // [batch_size, output_size]

        for (Tensor input : inputs) {
            int batchSize = input.shape()[0];

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
        }
    }

    protected void predictBatch(Pair<Tensor[], Tensor> batch, AtomicReference<Double> totalError) {
        Tensor[] inputs = batch.first();
        Tensor targets = batch.second();

        Tensor outputs = predict(inputs).cpu();
        int batchSize = outputs.shape()[0];

        for (int i = 0; i < batchSize; i++) {
            Range range = new Range(i, i + 1);

            Tensor output = outputs.slice(range).vector();
            Tensor target = targets.slice(range).vector();

            double loss = lossFunction.calculate(target, output);
            totalError.updateAndGet(v -> v + loss);
        }
    }

    protected Tensor validateInputs(Tensor... inputs) {
        if (inputs.length > 1 || inputs.length == 0) {
            throw new IllegalArgumentException("Input array must contain one element. Got: " + inputs.length);
        }

        Tensor input = inputs[0];

        if (input == null || input.dimension() == 0) {
            throw new IllegalArgumentException("Input is either null or has dimension of 0!");
        }

        if (input.dimension() < 2) {
            // Shape: [batch_size, input_size]
            input = input.reshape(1, input.elements());
        }

        int[] shape = input.shape();
        Layer inputLayer = layers.getFirst();

        if (!inputLayer.validateInput(input)) {
            throw new IllegalArgumentException(
                "Input shape mismatch! Expected " + inputLayer.size() + " but got " + shape[shape.length - 1]
            );
        }

        return input;
    }

    protected void printEvaluation(int step, int epoches, ListDataSource testSource) {
        EvaluationResult result = evaluate(testSource.clone());

        String lossMsg = "Loss: " + MAGENTA + "%.4f" + RESET;
        String accuracyMsg = "Accuracy: " + LIGHT_BLUE + "%.2f%%" + RESET;
        String f1ScoreMsg = "F1-Score: " + LIGHT_GREEN + "%.2f%%" + RESET;

        String prefix = "Epoch " + LIGHT_YELLOW + "%s" + WHITE + "/" + LIGHT_YELLOW + "%s " + WHITE;
        String message = prefix + lossMsg + " | " + accuracyMsg + " | " + f1ScoreMsg + "\n";
        String formatted = message.formatted(step, epoches, result.loss(), result.accuracy() * 100, result.f1Score() * 100);

        training.info(formatted);
    }

    private void printProgress(ListDataSource source, int epoch, int epoches, int batch, double tookMs) {
        String barChar = Commons.getHeaderChar();

        int progressBarLength = 25;
        int total = source.batches();

        double percentage = (double) batch / total;
        double tookInSeconds = tookMs / 1000;

        String timeStr = Commons.formatDuration(tookInSeconds);

        String intro = "Epoch " + LIGHT_YELLOW + "%s" + WHITE + "/" + LIGHT_YELLOW + "%s";
        String batchesMsg = LIGHT_BLUE + "%s" + WHITE + "/" + LIGHT_BLUE + "%s " + WHITE + "batches";
        String time = GRAY + " [%s/batch]" + RESET;

        String progressBar = " " + LIGHT_GREEN + Commons.createProgressBar(
            percentage,
            progressBarLength,
            barChar,
            RESET + barChar
        ) + " ";

        String message = String.format(intro + progressBar + batchesMsg + time,
            epoch, epoches, batch, total, timeStr);

        training.info(message);
    }

    @Override
    public Model add(Layer layer) {
        layers.add(layer);
        flattened.add(layer);
        return this;
    }

    @Override
    public Model add(int index, Layer layer) {
        layers.add(index, layer);
        flattened.add(index, layer);
        return this;
    }

    @Override
    public void fit(ListDataSource train, ListDataSource validation, int epoches, int evaluateEvery) {
        for (int epoch = 1; epoch <= epoches; epoch++) {
            int finalEpoch = epoch;

            AtomicReference<Double> totalForBatch = new AtomicReference<>(0.0);

            backPropagation.iteration(train, (batch, took) -> {
                totalForBatch.set(totalForBatch.get() + took);
                double average = totalForBatch.get() / batch;

                if (Brain4J.logging()) {
                    printProgress(train, finalEpoch, epoches, batch, average);
                }
            });

            if (epoch % evaluateEvery == 0) {
                if (Brain4J.logging()) {
                    System.out.println();
                }
                
                printEvaluation(epoch, epoches, validation);
            }
        }
    }
    
    @Override
    public Tensor predict(StatesCache cache, boolean training, Tensor... inputs) {
        Tensor input = validateInputs(inputs);
        Tensor result = input.to(device).withGrad();

        if (device != null) {
            GpuContextHandler.updateQueue(device, cache.commandQueue());
        }

        for (int i = 0; i < flattened.size(); i++) {
            Layer layer = flattenedAt(i);

            if (layer == null) {
                throw new IllegalStateException("Layer at index " + i + " is null!");
            }

            result = layer.forward(new ForwardContext(cache, result, i, training));
        }

        if (!training && device != null) {
            GpuContextHandler.closeQueue(device);
        }

        return result;
    }

    @Override
    public void backpropagate(StatesCache cache, Tensor outputs, Tensor targets) {
        int count = flattened.size() - 1;

        Layer last = flattened.getLast();
        last.computeLoss(cache, targets, outputs, lossFunction);

        for (int l = count; l >= 0; l--) {
            Layer layer = flattened.get(l);

            if (layer.skipPropagate()) continue;

            layer.backward(updater, optimizer, l);
        }
    }

    @Override
    public void updateWeights(Consumer<Layer> callback) {
        for (Layer layer : flattened) {
            callback.accept(layer);
        }
    }

    @Override
    public EvaluationResult evaluate(ListDataSource dataSource) {
        int classes = Math.max(2, dataSource.samples().getFirst().label().elements());
        Map<Integer, Tensor> classifications = new HashMap<>();

        for (int i = 0; i < classes; i++) {
            classifications.put(i, Tensors.zeros(classes));
        }

        AtomicReference<Double> totalLoss = new AtomicReference<>(0.0);

        Pair<Tensor[], Tensor> all = dataSource.allData();
        makeEvaluation(all, classifications, totalLoss);

        return new EvaluationResult(totalLoss.get() / dataSource.size(), classes, classifications);
    }

    @Override
    public double loss(ListDataSource dataSource) {
        AtomicReference<Double> totalError = new AtomicReference<>(0.0);

        Pair<Tensor[], Tensor> all = dataSource.allData();
        predictBatch(all, totalError);
        
        return totalError.get() / dataSource.size();
    }
    
    @Override
    public Model to(Device device) {
        this.device = device;

        GpuTensor.initKernels(device);
        Brain4J.initKernels(device);

        for (Layer layer : flattened) {
            layer.toDevice(device);
        }

        return this;
    }

    @Override
    public Device device() {
        return device;
    }

    @Override
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

    @Override
    public void summary() {
        if (updater == null || optimizer == null) {
            throw new IllegalStateException("The network is not compiled! Make sure to call compile() before.");
        }

        StringBuilder stats = new StringBuilder();
        DecimalFormat format = new DecimalFormat("#,###");

        String pattern = "%-7s %-20s %-12s %-15s %-15s\n";
        String divider = Commons.getHeader(" Architecture ", Commons.getHeaderChar());

        stats.append(divider);
        stats.append(pattern.formatted("Index", "Layer Type", "Parameters", "Shape", "Activation")).append("\n");

        AtomicLong totalWeights = new AtomicLong(0);
        AtomicLong totalBiases = new AtomicLong(0);

        append(pattern, stats, format, totalWeights, totalBiases);

        long weightsCount = totalWeights.get();
        long biasesCount = totalBiases.get();

        long params = weightsCount + biasesCount;

        String parameters = format.format(params);
        String weights = format.format(totalWeights);
        String biases = format.format(totalBiases);

        byte floatSize = Float.BYTES; // 4 bytes
        String sizeOfParams = Commons.formatNumber(params * floatSize);
        String sizeOfWeights = Commons.formatNumber(weightsCount * floatSize);
        String sizeOfBiases = Commons.formatNumber(biasesCount * floatSize);

        stats.append(Commons.getHeader(" Recap ", Commons.getHeaderChar()));
        stats.append("Total weights: %s (%s)\n".formatted(weights, sizeOfWeights));
        stats.append("Total biases: %s (%s)\n".formatted(biases, sizeOfBiases));
        stats.append("Total parameters: %s (%s)\n".formatted(parameters, sizeOfParams));
        stats.append(Commons.getHeader("", Commons.getHeaderChar()));

        Arrays.stream(stats.toString().split("\n")).forEach(logger::info);
    }

    private void append(
        String pattern,
        StringBuilder builder,
        DecimalFormat format,
        AtomicLong totalWeights,
        AtomicLong totalBiases
    ) {
        for (int i = 0; i < flattened.size(); i++) {
            Layer layer = flattenedAt(i);
            String layerType = layer.getClass().getSimpleName();

            int neurons = layer.size();
            int weights = layer.totalWeights() + layer.totalBiases();

            Tensor weightsTensor = layer.weights();

            String formatWeights = weights == 0 ? "-" : format.format(weights);
            String shape = weightsTensor == null
                    ? "[" + neurons + "]"
                    : Arrays.toString(weightsTensor.shape());

            builder.append(pattern.formatted(i, layerType, formatWeights, shape, layer.activation().name()));

            totalWeights.addAndGet(weights);
            totalBiases.addAndGet(neurons);
        }
    }

    @Override
    public Layer layerAt(int index) {
        return layers.get(index);
    }

    @Override
    public Layer flattenedAt(int index) {
        return flattened.get(index);
    }

    @Override
    public Layer connect(Layer previous) {
        int size = size();

        for (int i = 0; i < size; i++) {
            Layer layer = layerAt(i);
            previous = layer.connect(previous);
        }

        int[] inputSizes = new int[size];

        for (int i = 0; i < size; i++) {
            inputSizes[i] = (i == 0) ? 0 : layerAt(i - 1).size();
        }

        IntStream.range(0, size).parallel().forEach(i -> {
            Layer layer = layerAt(i);

            int input = inputSizes[i];
            int output = layer.size();

            Random localRandom = Random.from(new SplittableRandom(seed + i));
            layer.initWeights(localRandom, input, output);
        });

        return previous;
    }

    @Override
    public Tensor forward(ForwardContext context) {
        Tensor pass = context.input();
        StatesCache cache = context.cache();
        boolean training = context.training();

        for (int i = 0; i < size(); i++) {
            Layer layer = layerAt(i);

            if (layer == null) {
                throw new IllegalStateException("Layer at index " + i + " is null!");
            }

            pass = layer.forward(new ForwardContext(cache, pass, i, training));
        }

        return pass;
    }

    @Override
    public void backward(Updater updater, Optimizer optimizer, int index) {
        for (int l = size() - 2; l >= 0; l--) {
            Layer layer = layerAt(l);

            if (layer.skipPropagate()) continue;

            layer.backward(updater, optimizer, index - l);
        }
    }

    @Override
    public void zeroGrad() {
        for (Layer layer : flattened) {
            layer.resetGrad();
        }
    }

    @Override
    public int size() {
        return layers.size();
    }

    @Override
    public List<Layer> layers() {
        return new ArrayList<>(layers);
    }

    @Override
    public List<Layer> flattened() {
        return new ArrayList<>(flattened);
    }

    @Override
    public Optimizer optimizer() {
        return optimizer;
    }

    @Override
    public Updater updater() {
        return updater;
    }

    @Override
    public LossFunction lossFunction() {
        return lossFunction;
    }

    /**
     * Returns the seed value used to initialize the random number generator.
     * @return the seed value
     */
    public long seed() {
        return seed;
    }

    /**
     * Updates the seed value used to initialize the random number generator.
     * @param seed the new seed value
     * @return the model instance
     */
    public Model setSeed(long seed) {
        this.seed = seed;
        return this;
    }

    @Override
    public Iterator<Layer> iterator() {
        return new Iterator<>() {
            private int currentIndex = 0;

            @Override
            public boolean hasNext() {
                return currentIndex < flattened.size();
            }

            @Override
            public Layer next() {
                return flattenedAt(currentIndex++);
            }
        };
    }
    
    public static Sequential.Builder newBuilder() {
        return new Sequential.Builder();
    }
    
    public static class Builder {
        
        protected final List<Layer> layers;
        
        protected Updater updater;
        protected Optimizer optimizer;
        protected LossFunction lossFunction;
        
        protected Builder() {
            this.layers = new ArrayList<>();
            this.updater = new StochasticUpdater();
        }
        
        public Builder addDense(int dimension, Activations activation) {
            return add(new DenseLayer(dimension, activation));
        }
        
        public Builder add(Layer layer)  {
            if (layer == null) {
                throw new NullPointerException("Layer cannot be null!");
            }
            
            this.layers.add(layer);
            return this;
        }
        
        public Builder setOptimizer(Optimizer optimizer) {
            if (updater == null) {
                throw new NullPointerException("Optimizer cannot be null!");
            }
            
            this.optimizer = optimizer;
            return this;
        }
        
        public Builder setUpdater(Updater updater) {
            if (updater == null) {
                throw new NullPointerException("Updater cannot be null!");
            }
            
            this.updater = updater;
            return this;
        }
        
        public Builder setLossFunction(LossFunction lossFunction) {
            if (updater == null) {
                throw new NullPointerException("Loss function cannot be null!");
            }
            
            this.lossFunction = lossFunction;
            return this;
        }
        
        public Sequential compile() {
            Sequential model = new Sequential(layers.toArray(new Layer[0]));
            
            if (optimizer == null || lossFunction == null) {
                throw new IllegalStateException("Optimizer and loss function are both null! Initialize them first.");
            }
            
            return (Sequential) model.compile(lossFunction, optimizer, updater);
        }
    }
}
