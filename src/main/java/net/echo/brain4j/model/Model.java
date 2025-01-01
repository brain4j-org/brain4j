package net.echo.brain4j.model;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.reflect.TypeToken;
import net.echo.brain4j.adapters.LayerAdapter;
import net.echo.brain4j.adapters.OptimizerAdapter;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.layer.impl.DropoutLayer;
import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.initialization.WeightInitialization;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.training.BackPropagation;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.training.optimizers.impl.GradientDescent;
import net.echo.brain4j.training.updater.Updater;
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
 * Represents a generic neural network model.
 */
public class Model {

    private static final OptimizerAdapter OPTIMIZER_ADAPTER = new OptimizerAdapter();
    private static final LayerAdapter LAYER_ADAPTER = new LayerAdapter();
    private static final Gson GSON = new Gson()
            .newBuilder()
            .setPrettyPrinting()
            .excludeFieldsWithoutExposeAnnotation()
            .registerTypeAdapter(DenseLayer.class, LAYER_ADAPTER)
            .registerTypeAdapter(DropoutLayer.class, LAYER_ADAPTER)
            .registerTypeAdapter(Adam.class, OPTIMIZER_ADAPTER)
            .registerTypeAdapter(GradientDescent.class, OPTIMIZER_ADAPTER)
            .create();

    protected List<Layer> layers;
    protected LossFunctions function;
    protected Optimizer optimizer;
    protected Updater updater;
    protected BackPropagation propagation;
    protected Random generator;
    private int seed;

    public Model(Layer... layers) {
        this.layers = new ArrayList<>(Arrays.asList(layers));
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

    private void connect(WeightInitialization type) {
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

            double bound = type.getInitializer().getBound(nIn, nOut);

            lastNormalLayer.connectAll(generator, layer, bound);
            lastNormalLayer = layer;
        }
    }

    /**
     * Initializes the model and layers.
     *
     * @param type initialization method
     * @param function loss function for error assessment
     * @param optimizer optimization algorithm for training
     * @param updater weights updating algorithm for training
     */
    public void compile(WeightInitialization type, LossFunctions function, Optimizer optimizer, Updater updater) {
        this.generator = new Random(seed);
        this.function = function;
        this.optimizer = optimizer;
        this.updater = updater;
        this.propagation = new BackPropagation(this, optimizer, updater);

        connect(type);

        this.optimizer.postInitialize();
        this.updater.postInitialize();
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

        for (DataRow row : set.getDataRows()) {
            Vector inputs = row.inputs();
            Vector targets = row.outputs();

            Vector outputs = predict(inputs);

            totalError += function.getFunction().calculate(targets, outputs);
        }

        return totalError;
    }

    /**
     * Predicts output for given input.
     *
     * @param input input data
     * @return predicted outputs
     */
    public Vector predict(Vector input) {
        Layer inputLayer = layers.getFirst();

        if (input.toArray().length != inputLayer.getNeurons().size()) {
            throw new IllegalArgumentException("Input size does not match model's input dimension! (Input != Expected) " +
                    input.toArray().length + " != " + inputLayer.getNeurons().size());
        }

        for (Layer layer : layers) {
            for (Neuron neuron : layer.getNeurons()) {
                neuron.setValue(0);
            }
        }

        for (int i = 0; i < input.toArray().length; i++) {
            inputLayer.getNeuronAt(i).setValue(input.get(i));
        }

        for (int l = 0; l < layers.size() - 1; l++) {
            Layer layer = layers.get(l);

            if (layer instanceof DropoutLayer) continue;

            Layer nextLayer = layers.get(l + 1);

            if (nextLayer instanceof DropoutLayer) {
                nextLayer = layers.get(l + 2);
            }

            for (Synapse synapse : layer.getSynapses()) {
                Neuron inputNeuron = synapse.getInputNeuron();
                Neuron outputNeuron = synapse.getOutputNeuron();

                outputNeuron.setValue(outputNeuron.getValue() + inputNeuron.getValue() * synapse.getWeight());
            }

            nextLayer.applyFunction(layer);
        }

        Layer outputLayer = layers.getLast();

        double[] output = new double[outputLayer.getNeurons().size()];

        for (int i = 0; i < output.length; i++) {
            output[i] = outputLayer.getNeuronAt(i).getValue();
        }

        return Vector.of(output);
    }

    /**
     * Gets the model's loss function.
     *
     * @return loss function
     */
    public LossFunction getLossFunction() {
        return function.getFunction();
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
     * Generates model statistics.
     *
     * @return model stats
     */
    public String getStats() {
        StringBuilder stats = new StringBuilder();
        stats.append(String.format("%-7s %-15s %-10s %-12s\n", "Index", "Layer name", "nIn, nOut", "TotalParams"));
        stats.append("================================================\n");

        int params = 0;

        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            Layer next = layers.get(Math.min(i, layers.size() - 1));

            if (next instanceof DropoutLayer) {
                next = layers.get(Math.min(i + 1, layers.size() - 1));
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
            this.function = LossFunctions.valueOf(parent.get("lossFunction").getAsString());

            Type listType = new TypeToken<ArrayList<Layer>>(){}.getType();

            this.layers = GSON.fromJson(parent.get("layers"), listType);

            connect(WeightInitialization.NORMAL);

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

        parent.addProperty("lossFunction", function.name());
        parent.add("optimizer", optimizerObject);

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
     * @param layers the layers to add
     */
    public void add(Layer... layers) {
        this.layers.addAll(Arrays.asList(layers));
    }
}
