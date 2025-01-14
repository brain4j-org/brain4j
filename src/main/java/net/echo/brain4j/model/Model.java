package net.echo.brain4j.model;

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
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.threading.NeuronCacheHolder;
import net.echo.brain4j.training.BackPropagation;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.training.optimizers.impl.AdamW;
import net.echo.brain4j.training.optimizers.impl.GradientDescent;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.training.updater.impl.BatchedUpdater;
import net.echo.brain4j.training.updater.impl.NormalUpdater;
import net.echo.brain4j.training.updater.impl.StochasticUpdater;
import net.echo.brain4j.utils.Vector;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Represents a feed forward neural network.
 */
public class Model {

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
            .registerTypeAdapter(BatchedUpdater.class, UPDATER_ADAPTER)
            .registerTypeAdapter(StochasticUpdater.class, UPDATER_ADAPTER)

            .create();

    protected List<Layer> layers;
    protected List<Vector[]> synapsesMatrices;

    protected LossFunctions lossFunction;
    protected Optimizer optimizer;
    protected Updater updater;
    protected BackPropagation propagation;
    protected WeightInit weightInit;

    protected Random generator;
    private int seed;

    public Model(Layer... layers) {
        this.layers = new ArrayList<>(Arrays.asList(layers));
        this.synapsesMatrices = new ArrayList<>();
    }

    private void connect(WeightInit weightInit) {
        Layer lastNormalLayer = layers.getFirst();

        for (Layer layer : layers) {
            if (layer instanceof DenseLayer denseLayer) {
                denseLayer.init(generator);
            }
        }

        for (int i = 1; i < layers.size(); i++) {
            Layer layer = layers.get(i);

            if (layer.getNeurons().isEmpty()) continue;

            int nIn = lastNormalLayer.getNeurons().size();
            int nOut = layer.getNeurons().size();

            double bound = weightInit.getInitializer().getBound(nIn, nOut);

            lastNormalLayer.connectAll(generator, layer, bound);
            lastNormalLayer = layer;
        }
    }

    /**
     * Initializes the model and layers.
     *
     * @param weightInit initialization method
     * @param function loss function for error assessment
     * @param optimizer optimization algorithm for training
     * @param updater weights updating algorithm for training
     */
    public void compile(WeightInit weightInit, LossFunctions function, Optimizer optimizer, Updater updater) {
        this.weightInit = weightInit;
        this.generator = new Random(seed);
        this.lossFunction = function;
        this.optimizer = optimizer;
        this.updater = updater;
        this.propagation = new BackPropagation(this, optimizer, updater);

        connect(weightInit);

        this.optimizer.postInitialize(this);
        this.updater.postInitialize(this);

        reloadMatrices();
    }

    /**
     * Trains the model for one epoch.
     *
     * @param set dataset for training
     */
    public void fit(DataSet set) {
        propagation.iterate(set);
    }

    /**
     * Evaluates the model on the given dataset.
     *
     * @param set dataset for testing
     * @return the error of the model
     */
    public double evaluate(DataSet set) {
        double totalError = 0.0;

        reloadMatrices();

        for (DataRow row : set.getData()) {
            Vector inputs = row.inputs();
            Vector targets = row.outputs();

            Vector outputs = predict(new NeuronCacheHolder(), inputs);

            totalError += lossFunction.getFunction().calculate(targets, outputs);
        }

        return totalError;
    }

    /**
     * Finds the next layer used for computations given the initial index
     *
     * @param index starting layer index
     * @return the next computation layer
     */
    public Layer getNextComputationLayer(int index) {
        Layer nextLayer = layers.get(index + 1);

        for (int j = 2; j < layers.size() && nextLayer instanceof DropoutLayer; j++) {
            nextLayer = layers.get(index + j);
        }

        return nextLayer;
    }

    /**
     * Reloads the synapse matrix for each layer.
     */
    public void reloadMatrices() {
        synapsesMatrices.clear();

        for (int i = 0; i < layers.size() - 1; i++) {
            Layer layer = layers.get(i);

            if (layer instanceof DropoutLayer) {
                synapsesMatrices.add(new Vector[]{});
                continue;
            }

            Layer nextLayer = getNextComputationLayer(i);

            List<Neuron> neurons = layer.getNeurons();
            List<Neuron> nextNeurons = nextLayer.getNeurons();

            Vector[] synapseMatrixLayer = recalculateSynapseMatrix(layer.getSynapses(), neurons.size(), nextNeurons.size());

            synapsesMatrices.add(synapseMatrixLayer);
        }
    }

    /**
     * Predicts the output for the given input.
     *
     * @param input input data as a vector, must have dimension equal to the model's input dimension
     * @return predicted outputs as a vector
     */
    public Vector predict(Vector input) {
        return predict(new NeuronCacheHolder(), input);
    }

    /**
     * Predicts output for given input.
     *
     * @param input input data as a vector, must have dimension equal to the model's input dimension
     * @return predicted outputs as a vector
     */
    public Vector predict(NeuronCacheHolder cacheHolder, Vector input) {
        Layer inputLayer = layers.getFirst();

        if (input.toArray().length != inputLayer.getNeurons().size()) {
            throw new IllegalArgumentException("Input size does not match model's input dimension! (Input != Expected) " +
                    input.toArray().length + " != " + inputLayer.getNeurons().size());
        }

        for (int i = 0; i < input.toArray().length; i++) {
            inputLayer.getNeuronAt(i).setValue(cacheHolder, input.get(i));
        }

        for (int l = 0; l < layers.size() - 1; l++) {
            Layer layer = layers.get(l);

            if (layer instanceof DropoutLayer) continue;

            Layer nextLayer = getNextComputationLayer(l);

            List<Neuron> neurons = layer.getNeurons();
            List<Neuron> nextNeurons = nextLayer.getNeurons();

            int inSize = neurons.size();
            int outSize = nextNeurons.size();

            Vector inputVector = new Vector(inSize);
            Vector[] synapseMatrix = synapsesMatrices.get(l);

            for (int i = 0; i < neurons.size(); i++) {
                inputVector.set(i, neurons.get(i).getValue(cacheHolder));
            }

            for (int i = 0; i < outSize; i++) {
                double value = synapseMatrix[i].weightedSum(inputVector);
                nextNeurons.get(i).setValue(cacheHolder, value);
            }

            nextLayer.applyFunction(cacheHolder, layer);
        }

        Layer outputLayer = layers.getLast();

        double[] output = new double[outputLayer.getNeurons().size()];

        for (int i = 0; i < output.length; i++) {
            output[i] = outputLayer.getNeuronAt(i).getValue(cacheHolder);
        }

        return Vector.of(output);
    }

    /**
     * Loads a model from a file.
     *
     * @param path path to model file
     */
    public void load(String path) {
        File file = new File(path);

        if (!file.exists()) {
            throw new IllegalArgumentException("File does not exist: " + path);
        }

        try {
            JsonObject parent = JsonParser.parseReader(new FileReader(file)).getAsJsonObject();

            this.optimizer = GSON.fromJson(parent.get("optimizer"), Optimizer.class);
            this.updater = GSON.fromJson(parent.get("updater"), Updater.class);
            this.lossFunction = LossFunctions.valueOf(parent.get("lossFunction").getAsString());

            Type listType = new TypeToken<ArrayList<Layer>>(){}.getType();

            this.layers = GSON.fromJson(parent.get("layers"), listType);
            this.weightInit = WeightInit.valueOf(parent.get("weightInit").getAsString());
            this.seed = parent.get("seed").getAsInt();
            this.generator = new Random(seed);

            connect(weightInit);

            double[][] weights = GSON.fromJson(parent.get("weights"), double[][].class);

            for (int i = 0; i < weights.length; i++) {
                double[] layerWeights = weights[i];
                Layer layer = layers.get(i);

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

        JsonObject optimizerObject = GSON.toJsonTree(optimizer).getAsJsonObject();
        JsonObject updaterObject = GSON.toJsonTree(updater).getAsJsonObject();

        parent.addProperty("seed", seed);
        parent.addProperty("weightInit", weightInit.name());
        parent.addProperty("lossFunction", lossFunction.name());

        parent.add("optimizer", optimizerObject);
        parent.add("updater", updaterObject);

        List<JsonObject> layerObjects = new ArrayList<>();

        for (Layer layer : layers) {
            layerObjects.add(GSON.toJsonTree(layer).getAsJsonObject());
        }

        parent.add("layers", GSON.toJsonTree(layerObjects).getAsJsonArray());

        double[][] weights = new double[layers.size()][];

        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
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
    public void add(Layer... layers) {
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
    public List<Layer> getLayers() {
        return layers;
    }

    /**
     * Gets the synapse weights as a matrix, this is used by the model to cache weights for faster computation.
     */
    public List<Vector[]> getSynapsesMatrices() {
        return synapsesMatrices;
    }

    /**
     * Generates model statistics.
     */
    public String getStats() {
        StringBuilder stats = new StringBuilder();

        stats.append(String.format("%-7s %-15s %-10s %-12s\n", "Index", "Layer name", "In, Out", "Total params"));
        stats.append("================================================\n");

        int params = 0;

        for (int i = 0; i < layers.size() - 1; i++) {
            Layer layer = layers.get(i);
            Layer next = layers.get(Math.min(i + 1, layers.size() - 1));

            for (int j = 1; j < layers.size() && next instanceof DropoutLayer; j++) {
                next = layers.get(Math.min(i + j, layers.size() - 1));
            }

            String layerType = layer.getClass().getSimpleName();

            int nIn = layer.getNeurons().size();
            int nOut = i == layers.size() - 1 ? 0 : next.getNeurons().size();

            int totalParams = layer.getTotalParams();

            String formatNin = layer instanceof DropoutLayer ? "-" : String.valueOf(nIn);
            String formatNout = layer instanceof DropoutLayer ? "-" : String.valueOf(nOut);

            stats.append(String.format("%-7d %-15s %-10s %-12d\n",
                    i, layerType, formatNin + ", " + formatNout, totalParams));

            params += totalParams;
        }

        stats.append("================================================\n");
        stats.append("Total parameters: ").append(params).append("\n");
        stats.append("================================================\n");
        return stats.toString();
    }

    /**
     * Sets the seed for the weights generation.
     *
     * @param seed the seed, may be any value
     */
    public void setSeed(int seed) {
        this.seed = seed;
    }

    /**
     * Returns the seed for the weights generation.
     *
     * @return the seed, generated randomly at first
     */
    public int getSeed() {
        return seed;
    }
}
