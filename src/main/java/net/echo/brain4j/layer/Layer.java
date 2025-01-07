package net.echo.brain4j.layer;

import com.google.gson.annotations.JsonAdapter;
import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.adapters.LayerAdapter;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.threading.NeuronCacheHolder;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@JsonAdapter(LayerAdapter.class)
public class Layer {

    private final List<Neuron> neurons = new ArrayList<>();
    private final List<Synapse> synapses = new ArrayList<>();
    private final Activations activation;

    public Layer(int input, Activations activation) {
        for (int i = 0; i < input; i++) {
            neurons.add(new Neuron());
        }

        this.activation = activation;
    }

    public void init(Random generator) {
        for (Neuron neuron : neurons) {
            neuron.setBias(2 * generator.nextDouble() - 1);
        }
    }

    public void connectAll(Random generator, Layer nextLayer, double bound) {
        for (Neuron neuron : neurons) {
            for (Neuron nextNeuron : nextLayer.neurons) {
                Synapse synapse = new Synapse(generator, neuron, nextNeuron, bound);
                neuron.addSynapse(synapse);

                synapses.add(synapse);
            }
        }
    }

    public void applyFunction(NeuronCacheHolder cacheHolder, Layer previous) {
        Activation function = activation.getFunction();

        function.apply(cacheHolder, neurons);
    }

    public void setInput(NeuronCacheHolder cacheHolder, Vector input) {
        if (input.size() != neurons.size()) {
            throw new IllegalArgumentException("Input size does not match model's input dimension! (Input != Expected) " +
                    input.size() + " != " + neurons.size());
        }

        for (int i = 0; i < input.size(); i++) {
            neurons.get(i).setValue(cacheHolder, input.get(i));
        }
    }

    public Vector getVector(NeuronCacheHolder cacheHolder) {
        Vector values = new Vector(neurons.size());

        for (int i = 0; i < neurons.size(); i++) {
            values.set(i, neurons.get(i).getValue(cacheHolder));
        }

        return values;
    }

    public void propagate(NeuronCacheHolder cacheHolder, Updater updater, Optimizer optimizer) {
        for (Synapse synapse : synapses) {
            Neuron inputNeuron = synapse.getInputNeuron();
            optimizer.applyGradientStep(cacheHolder, updater, this, inputNeuron, synapse);
        }
    }

    public List<Neuron> getNeurons() {
        return neurons;
    }

    public List<Synapse> getSynapses() {
        return synapses;
    }

    public Activations getActivation() {
        return activation;
    }

    public Neuron getNeuronAt(int i) {
        return neurons.get(i);
    }

    public int getTotalParams() {
        return synapses.size();
    }
}
