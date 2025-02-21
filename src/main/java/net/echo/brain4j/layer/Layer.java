package net.echo.brain4j.layer;

import com.google.common.base.Preconditions;
import com.google.gson.annotations.JsonAdapter;
import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.adapters.LayerAdapter;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.utils.MLUtils;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@JsonAdapter(LayerAdapter.class)
public abstract class Layer {

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

    public boolean isConvolutional() {
        return false;
    }

    public void init(Random generator) {
        for (Neuron neuron : this.neurons) {
            neuron.setBias(2 * generator.nextDouble() - 1);
        }
    }

    public void connectAll(Random generator, Layer nextLayer, double bound) {
        for (Neuron neuron : this.neurons) {
            for (Neuron nextNeuron : nextLayer.getNeurons()) {
                Synapse synapse = new Synapse(generator, neuron, nextNeuron, bound);
                neuron.addSynapse(synapse);

                synapses.add(synapse);
            }
        }
    }

    public void applyFunction(StatesCache cacheHolder, Layer previous) {
        Activation function = this.activation.getFunction();
        function.apply(cacheHolder, this.neurons);
    }

    public void setInput(StatesCache cacheHolder, Vector input) {
        Preconditions.checkState(input.size() == this.neurons.size(), "Input size does not match!" +
                " (Input != Expected) " + input.size() + " != " + this.neurons.size());

        for (int i = 0; i < input.size(); i++) {
            this.neurons.get(i).setValue(cacheHolder, input.get(i));
        }
    }

    public void propagate(StatesCache cacheHolder, Layer previous, Updater updater) {
        for (Neuron neuron : this.neurons) {
            double value = neuron.getValue(cacheHolder);
            double derivative = this.activation.getFunction().getDerivative(value);

            for (Synapse synapse : neuron.getSynapses()) {
                double weightChange = calculateGradient(cacheHolder, synapse, derivative);
                updater.acknowledgeChange(cacheHolder, synapse, weightChange);
            }
        }
    }

    public double calculateGradient(StatesCache cacheHolder, Synapse synapse, double derivative) {
        Neuron input = synapse.getInputNeuron();
        Neuron output = synapse.getOutputNeuron();

        double error = MLUtils.clipGradient(synapse.getWeight() * output.getDelta(cacheHolder));
        double delta = MLUtils.clipGradient(error * derivative);

        input.setDelta(cacheHolder, delta);

        return MLUtils.clipGradient(delta * input.getValue(cacheHolder));
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

    public int size() {
        return neurons.size();
    }

    public void forward(StatesCache cache, Vector[] synapseMatrix, Layer nextLayer) {
        List<Neuron> nextNeurons = nextLayer.getNeurons();

        int inSize = this.neurons.size();
        int outSize = nextNeurons.size();

        Vector inputVector = new Vector(inSize);

        for (int i = 0; i < inSize; i++) {
            inputVector.set(i, neurons.get(i).getValue(cache));
        }

        for (int i = 0; i < outSize; i++) {
            double value = synapseMatrix[i].weightedSum(inputVector);
            nextNeurons.get(i).setValue(cache, value);
        }

        nextLayer.applyFunction(cache, this);
    }
}
