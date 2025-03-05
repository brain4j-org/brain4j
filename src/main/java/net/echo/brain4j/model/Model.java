package net.echo.brain4j.model;

import com.google.common.base.Preconditions;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.reflect.TypeToken;
import net.echo.brain4j.adapters.LayerAdapter;
import net.echo.brain4j.adapters.OptimizerAdapter;
import net.echo.brain4j.adapters.UpdaterAdapter;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.DropoutLayer;
import net.echo.brain4j.layer.impl.LayerNorm;
import net.echo.brain4j.layer.impl.convolution.ConvLayer;
import net.echo.brain4j.layer.impl.convolution.FlattenLayer;
import net.echo.brain4j.layer.impl.convolution.InputLayer;
import net.echo.brain4j.layer.impl.convolution.PoolingLayer;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.structure.cache.Parameters;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.training.optimizers.impl.AdamW;
import net.echo.brain4j.training.optimizers.impl.GradientDescent;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.training.updater.impl.NormalUpdater;
import net.echo.brain4j.training.updater.impl.StochasticUpdater;
import net.echo.brain4j.utils.DataSet;
import net.echo.brain4j.utils.MLUtils;
import net.echo.brain4j.utils.Vector;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.lang.reflect.Type;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Represents a feed forward neural network.
 */
public abstract class Model<R, I, O> {

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

    protected List<Layer<?, ?>> layers;
    protected LossFunctions lossFunction;
    protected Optimizer optimizer;
    protected Updater updater;
    protected WeightInit weightInit;

    protected Random generator;
    private int seed;

    public Model(Layer<?, ?>... layers) {
        this.layers = new ArrayList<>(Arrays.asList(layers));

        if (this.layers.isEmpty()) return;

        validateLayers();
    }

    private void validateLayers() {
        boolean isInput = layers.getFirst() instanceof InputLayer;
        boolean hasConv = false;

        for (Layer<?, ?> layer : layers) {
            if (layer instanceof ConvLayer || layer instanceof PoolingLayer || layer instanceof FlattenLayer) {
                hasConv = true;
                break;
            }
        }

        if (isInput && !hasConv) throw new IllegalArgumentException("Cannot use the InputLayer outside of a convolutional model!");
        if (!isInput && hasConv) throw new IllegalArgumentException("Cannot use a convolutional layer without an InputLayer!");
    }

    public void connect(WeightInit weightInit, boolean update) {
        Layer<?, ?> lastNormalLayer = layers.getFirst();

        for (Layer<?, ?> layer : layers) {
            if (layer instanceof DenseLayer denseLayer && update) {
                denseLayer.init(generator);
            }
        }

        for (int i = 1; i < layers.size(); i++) {
            Layer<?, ?> layer = layers.get(i);

            if (layer.getNeurons().isEmpty() && !layer.isConvolutional()) continue;

            int nIn = lastNormalLayer.size();
            int nOut = layer.size();

            double bound = weightInit.getInitializer().getBound(nIn, nOut);

            lastNormalLayer.connectAll(generator, layer, bound);
            lastNormalLayer = layer;
        }
    }

    /**
     * Initializes the model and layers with default values.
     *
     * @param function loss function for evaluation and training
     * @param optimizer optimization algorithm to use
     * @return the instance of the model
     */
    public Model<R, I, O> compile(LossFunctions function, Optimizer optimizer) {
        return compile(WeightInit.UNIFORM_XAVIER, function, optimizer, new StochasticUpdater());
    }

    /**
     * Initializes the model and layers with default values.
     *
     * @param weightInit initialization method for weights
     * @param function loss function for evaluation and training
     * @param optimizer optimization algorithm to use
     * @param updater weights updating algorithm for training
     */
    public Model<R, I, O> compile(WeightInit weightInit, LossFunctions function, Optimizer optimizer, Updater updater) {
        this.weightInit = weightInit;
        this.generator = new Random(this.seed);
        this.lossFunction = function;
        this.optimizer = optimizer;
        this.updater = updater;
        return this;
    }

    /**
     * Evaluates the model on the given dataset.
     *
     * @param set dataset for testing
     * @return the error of the model
     */
    public abstract double evaluate(DataSet<R> set);

    /**
     * Finds the next layer used for computations given the initial index
     *
     * @param index starting layer index
     * @return the next computation layer
     */
    public Layer<?, ?> getNextComputationLayer(int index) {
        Layer<?, ?> nextLayer = layers.get(index + 1);

        for (int j = 2; j < layers.size() && nextLayer instanceof DropoutLayer; j++) {
            nextLayer = layers.get(index + j);
        }

        return nextLayer;
    }

    /**
     * Trains the model for one epoch.
     *
     * @param dataSet dataset for training
     */
    public abstract void fit(DataSet<R> dataSet);

    /**
     * Predicts the output for the given input.
     *
     * @param input input data as a vector, must have dimension equal to the model's input dimension
     * @return predicted outputs as a vector
     */
    public abstract O predict(I input);

    /**
     * Predicts output for given input.
     *
     * @param input input data as a vector, must have dimension equal to the model's input dimension
     * @param cache cache used to store neuron states
     * @return predicted outputs as a vector
     */
    public abstract O predict(StatesCache cache, I input);

    /**
     * Loads a model from a file.
     *
     * @param path path to model file
     */
    public void load(String path) {
        File file = new File(path);

        Preconditions.checkState(file.exists(), "File does not exist: " + path);

        try {
            JsonObject parent = JsonParser.parseReader(new FileReader(file)).getAsJsonObject();

            this.optimizer = GSON.fromJson(parent.get("optimizer"), Optimizer.class);
            this.updater = GSON.fromJson(parent.get("updater"), Updater.class);
            this.lossFunction = LossFunctions.valueOf(parent.get("lossFunction").getAsString());

            Type listType = new TypeToken<ArrayList<Layer<?, ?>>>(){}.getType();

            this.weightInit = WeightInit.valueOf(parent.get("weightInit").getAsString());
            this.seed = parent.get("seed").getAsInt();
            this.generator = new Random(this.seed);
            this.layers = GSON.fromJson(parent.get("layers"), listType);

            Parameters.TOTAL_SYNAPSES = 0;
            connect(this.weightInit, false);

            double[][] biases = GSON.fromJson(parent.get("biases"), double[][].class);

            for (int i = 0; i < biases.length; i++) {
                double[] layerBiases = biases[i];
                Layer<?, ?> layer = this.layers.get(i);

                for (int j = 0; j < layerBiases.length; j++) {
                    layer.getNeuronAt(j).setBias(layerBiases[j]);
                }
            }

            double[][] weights = GSON.fromJson(parent.get("weights"), double[][].class);

            for (int i = 0; i < weights.length; i++) {
                double[] layerWeights = weights[i];
                Layer<?, ?> layer = this.layers.get(i);

                for (int j = 0; j < layerWeights.length; j++) {
                    layer.getSynapses().get(j).setWeight(layerWeights[j]);
                }
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Saves the model to a file.
     *
     * @param path path to save model
     */
    public void save(String path) {
        File file = new File(path);

        JsonObject parent = new JsonObject();

        JsonObject optimizerObject = GSON.toJsonTree(this.optimizer).getAsJsonObject();
        JsonObject updaterObject = GSON.toJsonTree(this.updater).getAsJsonObject();

        parent.addProperty("seed", this.seed);
        parent.addProperty("weightInit", this.weightInit.name());
        parent.addProperty("lossFunction", this.lossFunction.name());

        parent.add("optimizer", optimizerObject);
        parent.add("updater", updaterObject);

        List<JsonObject> layerObjects = new ArrayList<>();

        for (Layer<?, ?> layer : this.layers) {
            layerObjects.add(GSON.toJsonTree(layer).getAsJsonObject());
        }

        parent.add("layers", GSON.toJsonTree(layerObjects).getAsJsonArray());

        double[][] biases = new double[this.layers.size()][];

        for (int i = 0; i < this.layers.size(); i++) {
            Layer<?, ?> layer = this.layers.get(i);
            biases[i] = new double[layer.getNeurons().size()];

            for (int j = 0; j < biases[i].length; j++) {
                biases[i][j] = layer.getNeuronAt(j).getBias();
            }
        }

        parent.add("biases", GSON.toJsonTree(biases));

        double[][] weights = new double[this.layers.size()][];

        for (int i = 0; i < this.layers.size(); i++) {
            Layer<?, ?> layer = this.layers.get(i);
            weights[i] = new double[layer.getSynapses().size()];

            for (int j = 0; j < layer.getSynapses().size(); j++) {
                Synapse synapse = layer.getSynapses().get(j);

                weights[i][j] = synapse.getWeight();
            }
        }

        parent.add("weights", GSON.toJsonTree(weights));

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
            GSON.toJson(parent, writer);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
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
     * Recalculates the synapse matrix, used to cache the synapse weights for faster computation.
     *
     * @param synapses list of synapses to cache
     * @param inSize input size of the vector
     * @param outSize output size of the vector
     *
     * @return the synapse matrix
     */
    public Vector[] recalculateSynapseMatrix(List<Synapse> synapses, int inSize, int outSize) {
        Vector[] synapseMatrix = new Vector[outSize];

        for (int i = 0; i < outSize; i++) {
            Vector vector = new Vector(inSize);
            synapseMatrix[i] = vector;

            for (int j = 0; j < inSize; j++) {
                Synapse synapse = synapses.get(j * outSize + i);
                vector.set(j, synapse.getWeight());
            }
        }

        return synapseMatrix;
    }

    /**
     * Generates model statistics.
     */
    public String getStats() {
        StringBuilder stats = new StringBuilder();

        String header = "==========================================================\n";

        stats.append(String.format("%-7s %-15s %-10s %-12s %-15s\n", "Index", "Layer", "Neurons", "Weights", "Activation"));
        stats.append(header);

        int totalSynapses = 0;
        int totalBiases = 0;

        for (int i = 0; i < this.layers.size(); i++) {
            Layer<?, ?> layer = this.layers.get(i);

            String layerType = layer.getClass().getSimpleName();

            int biases = layer.size();
            int synapses = layer.getTotalParams();

            String formatNin = layer instanceof DropoutLayer ? "-" : String.valueOf(biases);

            stats.append(String.format("%-7d %-15s %-10s %-12d %-15s\n",
                    i, layerType, formatNin, synapses, layer.getActivation().name()));

            totalSynapses += synapses;
            totalBiases += biases;
        }

        int params = totalSynapses + totalBiases;

        DecimalFormat format = new DecimalFormat("#,###");

        String parameters = format.format(params);
        String formatted = MLUtils.formatNumber(params * 4); // 4 = float size in bytes
        String actual = MLUtils.formatNumber(totalSynapses * 32);

        stats.append(header);
        stats.append("Total parameters: ").append(parameters).append(" (").append(formatted).append(")\n");
        stats.append("Expected memory usage: ").append(actual).append("\n");
        stats.append(header);

        return stats.toString();
    }

    /**
     * Gets the model's loss function.
     */
    public LossFunction getLossFunction() {
        return lossFunction.getFunction();
    }

    /**
     * Gets the model's optimizer.
     */
    public Optimizer getOptimizer() {
        return optimizer;
    }

    /**
     * Gets the model's weights updater.
     */
    public Updater getUpdater() {
        return updater;
    }

    /**
     * Gets the weight initialization technique used.
     */
    public WeightInit getWeightInit() {
        return weightInit;
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
    }

    /**
     * Returns the seed for the weight generation.
     *
     * @return the current seed
     */
    public int getSeed() {
        return seed;
    }
}
