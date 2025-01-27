package net.echo.brain4j.layer;

import com.google.gson.annotations.JsonAdapter;
import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.adapters.LayerAdapter;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.structure.StatesCache;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.utils.MLUtils;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@JsonAdapter(LayerAdapter.class)
public class Layer {

    protected final List<Neuron> neurons = new ArrayList<>();
    protected final List<Synapse> synapses = new ArrayList<>();
    protected final Activations activation;

    public Layer(int input, Activations activation) {
        for (int i = 0; i < input; i++) {
            this.neurons.add(new Neuron());
        }

        this.activation = activation;
    }

    public boolean canPropagate() {
        return true;
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

    public void applyFunction(StatesCache cacheHolder, Layer previous) {
        Activation function = activation.getFunction();

        function.apply(cacheHolder, neurons);
    }

    public void setInput(StatesCache cacheHolder, Vector input) {
        if (input.size() != neurons.size()) {
            throw new IllegalArgumentException("Input size does not match model's input dimension! (Input != Expected) " +
                    input.size() + " != " + neurons.size());
        }

        for (int i = 0; i < input.size(); i++) {
            neurons.get(i).setValue(cacheHolder, input.get(i));
        }
    }

    public Vector getVector(StatesCache cacheHolder) {
        Vector values = new Vector(neurons.size());

        for (int i = 0; i < neurons.size(); i++) {
            values.set(i, neurons.get(i).getValue(cacheHolder));
        }

        return values;
    }

    public void propagate(StatesCache cacheHolder, Layer previous, Updater updater) {
        Vector neuronValues = getVector(cacheHolder);
        Vector derivatives = activation.getFunction().getDerivative(neuronValues);

        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);

            for (Synapse synapse : neuron.getSynapses()) {
                double weightChange = calculateGradient(cacheHolder, synapse, derivatives, i);
                updater.acknowledgeChange(synapse, weightChange);
            }
        }
    }

    public double calculateGradient(StatesCache cacheHolder, Synapse synapse, Vector derivatives, int index) {
        Neuron neuron = synapse.getInputNeuron();
        double derivative = derivatives.get(index);

        double error = MLUtils.clipGradient(synapse.getWeight() * synapse.getOutputNeuron().getDelta(cacheHolder));
        double delta = MLUtils.clipGradient(error * derivative);

        neuron.setDelta(cacheHolder, delta);

        return MLUtils.clipGradient(delta * synapse.getInputNeuron().getValue(cacheHolder));
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
