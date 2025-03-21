package net.echo.brain4j.layer;

import com.google.common.base.Preconditions;
import com.google.gson.annotations.JsonAdapter;
import net.echo.math4j.BrainUtils;
import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.adapters.Adapter;
import net.echo.brain4j.adapters.json.LayerAdapter;
import net.echo.brain4j.loss.LossFunction;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.structure.cache.Parameters;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.math4j.math.vector.Vector;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@JsonAdapter(LayerAdapter.class)
public abstract class Layer<I, O> implements Adapter {


    protected WeightInitializer weightInit;
    protected LossFunction lossFunction;
    protected Optimizer optimizer;
    protected Updater updater;
    protected Layer<?, ?> nextLayer;

    protected List<Synapse> synapses;
    protected List<Neuron> neurons;
    protected Activation activation;
    protected int id;

    public Layer() {
        this(Activations.LINEAR);
    }

    public Layer(Activation activation) {
        this(0, activation);
    }

    public Layer(Activations activation) {
        this(0, activation);
    }

    public Layer(int input, Activations activation) {
        this(input, activation.getFunction());
    }

    public Layer(int input, Activation activation) {
        this.id = Parameters.TOTAL_LAYERS++;
        this.synapses = new ArrayList<>();
        this.neurons = new ArrayList<>();
        this.activation = activation;

        for (int i = 0; i < input; i++) {
            neurons.add(new Neuron());
        }
    }

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        stream.writeInt(id);
        stream.writeInt(neurons.size());
        stream.writeUTF(activation.getClass().getName());
    }

    @Override
    public void deserialize(DataInputStream stream) throws Exception {
        this.id = stream.readInt();

        int totalNeurons = stream.readInt();

        for (int i = 0; i < totalNeurons; i++) {
            neurons.add(new Neuron());
        }

        String activationClassPath = stream.readUTF();
        Class<?> activationClass = Class.forName(activationClassPath);

        this.activation = (Activation) activationClass.getDeclaredConstructor().newInstance();
    }

    public boolean canPropagate() {
        return true;
    }

    public boolean isConvolutional() {
        return false;
    }

    public void compile(WeightInitializer weightInit, LossFunction lossFunction, Optimizer optimizer, Updater updater) {
        this.weightInit = weightInit;
        this.lossFunction = lossFunction;
        this.optimizer = optimizer;
        this.updater = updater;
    }

    public void init(Random generator) {
        neurons.forEach(neuron -> neuron.setBias(2 * generator.nextDouble() - 1));
    }

    public void connect(Random generator, Layer<?, ?> nextLayer, double bound) {
        this.nextLayer = nextLayer;

        for (Neuron neuron : neurons) {
            for (Neuron nextNeuron : nextLayer.getNeurons()) {
                Synapse synapse = new Synapse(generator, neuron, nextNeuron, bound);
                synapses.add(synapse);
            }
        }
    }

    public O forward(StatesCache cache, Layer<?, ?> lastLayer, I input) {
        throw new UnsupportedOperationException("Not implemented for this class.");
    }

    public void updateWeights(Tensor weights) {
        throw new UnsupportedOperationException("Not implemented for this class.");
    }

    public void applyFunction(StatesCache cache, Layer<?, ?> previous) {
        activation.apply(cache, neurons);
    }

    public void setInput(StatesCache cache, Vector input) {
        Preconditions.checkState(input.size() == neurons.size(), "Input size does not match!" +
                " (Input != Expected) " + input.size() + " != " + neurons.size());

        for (int i = 0; i < input.size(); i++) {
            neurons.get(i).setValue(cache, input.get(i));
        }
    }

    public void propagate(StatesCache cache, Layer<?, ?> previous) {
        throw new UnsupportedOperationException("Not implemented for this class.");
    }

    public float calculateGradient(StatesCache cacheHolder, Synapse synapse, double derivative) {
        Neuron input = synapse.getInputNeuron();
        Neuron output = synapse.getOutputNeuron();

        float delta = output.getDelta(cacheHolder);
        float error = BrainUtils.clipGradient(synapse.getWeight() * delta * derivative);

        input.setDelta(cacheHolder, error);

        return BrainUtils.clipGradient(error * input.getValue(cacheHolder));
    }

    public List<Neuron> getNeurons() {
        return neurons;
    }

    public List<Synapse> getSynapses() {
        return synapses;
    }

    public Activation getActivation() {
        return activation;
    }

    public Neuron getNeuronAt(int i) {
        return neurons.get(i);
    }

    public int getTotalParams() {
        return synapses.size();
    }

    public int getTotalNeurons() {
        return neurons.size();
    }

    public int getId() {
        return id;
    }
}
