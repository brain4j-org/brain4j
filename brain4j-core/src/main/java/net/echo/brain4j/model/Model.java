package net.echo.brain4j.model;

import com.google.gson.Gson;
import net.echo.brain4j.adapters.Adapter;
import net.echo.brain4j.adapters.ModernAdapter;
import net.echo.brain4j.adapters.json.LayerAdapter;
import net.echo.brain4j.adapters.json.OptimizerAdapter;
import net.echo.brain4j.adapters.json.UpdaterAdapter;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.DropoutLayer;
import net.echo.brain4j.layer.impl.LayerNorm;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.model.impl.Transformer;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.BackPropagation;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.evaluation.EvaluationResult;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.training.optimizers.impl.AdamW;
import net.echo.brain4j.training.optimizers.impl.GradientDescent;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.training.updater.impl.NormalUpdater;
import net.echo.brain4j.training.updater.impl.StochasticUpdater;
import net.echo.brain4j.transformers.TransformerEncoder;
import net.echo.math4j.BrainUtils;
import net.echo.math4j.DataSet;
import net.echo.math4j.math.tensor.Tensor;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Represents an abstract neural network model.
 *
 * @param <R> the data example type for training
 * @param <I> the input type of this model
 * @param <O> the output type of this model
 *
 * @see Sequential Sequential
 * @see Transformer Transformer
 */
public abstract class Model implements Adapter {

    private static final OptimizerAdapter OPTIMIZER_ADAPTER = new OptimizerAdapter();
    private static final UpdaterAdapter UPDATER_ADAPTER = new UpdaterAdapter();
    private static final LayerAdapter LAYER_ADAPTER = new LayerAdapter();
    private static final Gson GSON = new Gson()
            .newBuilder()
            .setPrettyPrinting()
            .excludeFieldsWithoutExposeAnnotation()

            .registerTypeAdapter(DenseLayer.class, LAYER_ADAPTER)
            .registerTypeAdapter(LayerNorm.class, LAYER_ADAPTER)
            .registerTypeAdapter(DropoutLayer.class, LAYER_ADAPTER)

            .registerTypeAdapter(Adam.class, OPTIMIZER_ADAPTER)
            .registerTypeAdapter(AdamW.class, OPTIMIZER_ADAPTER)
            .registerTypeAdapter(GradientDescent.class, OPTIMIZER_ADAPTER)

            .registerTypeAdapter(NormalUpdater.class, UPDATER_ADAPTER)
            .registerTypeAdapter(StochasticUpdater.class, UPDATER_ADAPTER)

            .create();

    protected BackPropagation propagation;
    protected List<Layer<?, ?>> layers;

    protected WeightInitializer weightInit;
    protected LossFunction lossFunction;
    protected Optimizer optimizer;
    protected Updater updater;

    protected Random generator;
    protected int seed;

    public Model(Layer<?, ?>... layers) {
        this.layers = new ArrayList<>(Arrays.asList(layers));
    }

    public void connect() {
        connect(weightInit, false);
    }

    public void connect(WeightInitializer weightInit, boolean update) {
        Layer<?, ?> lastNormalLayer = layers.getFirst();

        for (Layer<?, ?> layer : layers) {
            layer.compile(weightInit, lossFunction, optimizer, updater);

            if (layer instanceof DenseLayer denseLayer && update) {
                denseLayer.init(generator);
            }
        }

        for (int i = 1; i < layers.size(); i++) {
            Layer<?, ?> layer = layers.get(i);

            if (layer.getTotalNeurons() == 0 && !layer.isConvolutional()) continue;

            int nIn = lastNormalLayer.getTotalNeurons();
            int nOut = layer.getTotalNeurons();

            double bound = weightInit.getBound(nIn, nOut);

            lastNormalLayer.connect(generator, layer, bound);
            lastNormalLayer = layer;
        }
    }

    /**
     * Evaluates the model on the given dataset.
     *
     * @param dataSet dataset for testing
     * @return the error of the model
     */
    public abstract double loss(DataSet<DataRow> dataSet);

    /**
     * Evaluates the model performance on the given dataset.
     *
     * @param dataSet dataset to evaluate
     * @return an evaluation result
     */
    public abstract EvaluationResult evaluate(DataSet<DataRow> dataSet);

    /**
     * Trains the model for one epoch.
     *
     * @param dataSet dataset for training
     */
    public abstract void fit(DataSet<DataRow> dataSet);

    /**
     * Predicts output for given input.
     *
     * @param input input data as a vector, must have dimension equal to the model's input dimension
     * @param cache cache used to store neuron states
     * @return predicted outputs as a vector
     */
    public abstract Tensor predict(StatesCache cache, Tensor input, boolean training);

    /**
     * Reloads the values inside the network.
     */
    public abstract void reloadWeights();

    /**
     * Trains the model for the given number of epoches.
     *
     * @param dataSet dataset for training
     * @param epoches number of epoches
     */
    public void fit(DataSet<DataRow> dataSet, int epoches) {
        for (int i = 0; i < epoches; i++) {
            fit(dataSet);
        }
    }

    /**
     * Predicts the output for the given input.
     *
     * @param input The input data.
     * @return The model's prediction.
     */
    public Tensor predict(Tensor input) {
        return predict(new StatesCache(), input, false);
    }

    /**
     * Initializes the model and layers with default values.
     *
     * @param function The loss function used while evaluation and training.
     * @param optimizer The gradient optimization algorithm to use.
     * @return The current instance of the model.
     */
    public Model compile(LossFunctions function, Optimizer optimizer) {
        return compile(function.getFunction(), optimizer);
    }

    /**
     * Initializes the model and layers with default values.
     *
     * @param function The loss function used while evaluation and training.
     * @param optimizer The gradient optimization algorithm to use.
     * @return The current instance of the model.
     */
    public Model compile(LossFunction function, Optimizer optimizer) {
        return compile(WeightInit.UNIFORM_XAVIER.getFunction(), function, optimizer, new StochasticUpdater());
    }

    /**
     * Initializes the model and layers with default values.
     *
     * @param initializer The initialization method for weights.
     * @param lossFunction The loss function to use while evaluating and training.
     * @param optimizer The gradient optimization algorithm to use when training.
     * @param updater The weights updating algorithm to use when training.
     * @return The current instance of the model.
     */
    public Model compile(WeightInitializer initializer, LossFunction lossFunction, Optimizer optimizer, Updater updater) {
        this.propagation = new BackPropagation(this, optimizer, updater);
        this.weightInit = initializer;
        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
        this.updater = updater;
        this.generator = new Random(this.seed);

        return this;
    }

    /**
     * Initializes the model and layers with default values.
     *
     * @param initializer The initialization method for weights.
     * @param lossFunction The loss function to use while evaluating and training.
     * @param optimizer The gradient optimization algorithm to use when training.
     * @param updater The weights updating algorithm to use when training.
     * @return The current instance of the model.
     */
    public Model compile(WeightInit initializer, LossFunctions lossFunction, Optimizer optimizer, Updater updater) {
        return compile(initializer.getFunction(), lossFunction.getFunction(), optimizer, updater);
    }

    /**
     * Loads a model from a file.
     *
     * @param path path to model file
     */
    public void load(String path) throws Exception {
        ModernAdapter.deserialize(path, this);
    }

    /**
     * Saves the model to a file.
     *
     * @param path path to save model
     */
    public void save(String path) throws Exception {
        ModernAdapter.serialize(path, this);
    }

    /**
     * Adds a layers to the network.
     *
     * @param layers a list of layers to add
     */
    public void add(Layer<?, ?>... layers) {
        this.layers.addAll(Arrays.asList(layers));
    }

    /**
     * Generates model statistics.
     */
    public String getStats() {

        StringBuilder stats = new StringBuilder();
        DecimalFormat format = new DecimalFormat("#,###");

        String header = "======================================================================\n";
        String pattern = "%-7s %-20s %-12s %-15s %-15s\n";
        String divider = BrainUtils.getHeader(" Architecture ");

        stats.append(divider);
        stats.append(String.format(pattern, "Index", "Layer", "Neurons", "Weights", "Activation"));
        stats.append("-".repeat(divider.length() - 1)).append("\n");

        int totalWeights = 0;
        int totalSynapses = 0;
        int totalBiases = 0;

        for (int i = 0; i < this.layers.size(); i++) {
            Layer<?, ?> layer = this.layers.get(i);

            String layerType = layer.getClass().getSimpleName();

            int neurons = layer.getTotalNeurons();
            int weights = layer.getTotalParams();

            String formatNeurons = layer instanceof DropoutLayer ? "-" : format.format(neurons);
            String formatWeights = format.format(weights);

            stats.append(String.format(pattern, i, layerType, formatNeurons, formatWeights, layer.getActivation().getName()));

            if (layer instanceof TransformerEncoder encoder) {
                totalSynapses += encoder.getFeedForwardSize();
            } else {
                totalSynapses += weights;
            }

            totalWeights += weights;
            totalBiases += neurons;
        }

        int params = totalWeights + totalBiases;

        String parameters = format.format(params);
        String synapses = format.format(totalSynapses);

        String formatted = BrainUtils.formatNumber(params * 4); // 4 = float size in bytes
        String actual = BrainUtils.formatNumber(totalSynapses * 32);

        stats.append(BrainUtils.getHeader(" Recap "));
        stats.append("Total parameters: ").append(parameters).append(" (").append(formatted).append(")\n");
        stats.append("Total synapses: ").append(synapses).append("\n");
        stats.append("Expected memory usage: ").append(actual).append("\n");
        stats.append(header);

        return stats.toString();
    }

    /**
     * Gets the model's loss function.
     */
    public LossFunction getLossFunction() {
        return lossFunction;
    }

    public void setLossFunction(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }

    /**
     * Gets the model's optimizer.
     */
    public Optimizer getOptimizer() {
        return optimizer;
    }

    public void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
    }

    /**
     * Gets the model's weights updater.
     */
    public Updater getUpdater() {
        return updater;
    }

    public void setUpdater(Updater updater) {
        this.updater = updater;
    }

    /**
     * Gets the weight initialization technique used.
     */
    public WeightInitializer getWeightInit() {
        return weightInit;
    }

    public void setWeightInit(WeightInitializer weightInit) {
        this.weightInit = weightInit;
    }

    /**
     * Gets the random generator used by the model.
     */
    public Random getGenerator() {
        return generator;
    }

    /**
     * Retrieves layers of the network.
     *
     * @return list of layers
     */
    public List<Layer<?, ?>> getLayers() {
        return layers;
    }

    /**
     * Sets the seed for the weight generation.
     *
     * @param seed the seed may be any value
     */
    public void setSeed(int seed) {
        this.seed = seed;
        this.generator = new Random(seed);
    }

    /**
     * Returns the seed for the weight generation.
     *
     * @return the current seed
     */
    public int getSeed() {
        return seed;
    }

    /**
     * Gets the amount of total neurons in the model.
     * @return the total number of neurons
     */
    public int getTotalNeurons() {
        int total = 0;

        for (Layer<?, ?> layer : layers) {
            total += layer.getTotalNeurons();
        }

        return total;
    }

    /**
     * Gets the total amount of weights in the model.
     * @return the total number of weights, usually the total amount of synapses
     */
    public int getTotalWeights() {
        int total = 0;

        for (Layer<?, ?> layer : layers) {
            total += layer.getTotalParams();
        }

        return total;
    }
}
