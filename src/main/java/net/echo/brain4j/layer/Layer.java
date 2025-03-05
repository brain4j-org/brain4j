package net.echo.brain4j.layer;

import com.google.common.base.Preconditions;
import com.google.gson.annotations.JsonAdapter;
import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.adapters.LayerAdapter;
import net.echo.brain4j.convolution.Kernel;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.structure.cache.Parameters;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;

import static net.echo.brain4j.utils.MLUtils.clipGradient;

@JsonAdapter(LayerAdapter.class)
public abstract class Layer<I, O> {

    protected final List<Neuron> neurons = new ArrayList<>();
    protected final List<Synapse> synapses = new ArrayList<>();
    protected final Activations activation;
    protected final Activation function;
    protected Layer<?, ?> nextLayer;

    public Layer(int input, Activations activation) {
        Parameters.TOTAL_LAYERS++;
        Stream.generate(Neuron::new).limit(input).forEach(neurons::add);

        this.activation = activation;
        this.function = activation.getFunction();
    }

    public boolean canPropagate() {
        return true;
    }

    public boolean isConvolutional() {
        return false;
    }

    public void init(Random generator) {
        neurons.forEach(neuron -> neuron.setBias(2 * generator.nextDouble() - 1));
    }

    public void connectAll(Random generator, Layer<?, ?> nextLayer, double bound) {
        this.nextLayer = nextLayer;

        for (Neuron neuron : neurons) {
            for (Neuron nextNeuron : nextLayer.getNeurons()) {
                Synapse synapse = new Synapse(generator, neuron, nextNeuron, bound);
                // neuron.addSynapse(synapse);

                synapses.add(synapse);
            }
        }
    }

    public O forward(StatesCache cache, Layer<?, ?> lastLayer, I input) {
        throw new UnsupportedOperationException("Not implemented for this class.");
    }

    public void updateWeights(Vector[] synapseMatrixLayer) {
        throw new UnsupportedOperationException("Not implemented for this class.");
    }

    public void applyFunction(StatesCache cacheHolder, Layer<?, ?> previous) {
        function.apply(cacheHolder, neurons);
    }

    public void setInput(StatesCache cacheHolder, Vector input) {
        Preconditions.checkState(input.size() == neurons.size(), "Input size does not match!" +
                " (Input != Expected) " + input.size() + " != " + neurons.size());

        for (int i = 0; i < input.size(); i++) {
            neurons.get(i).setValue(cacheHolder, input.get(i));
        }
    }

    public void propagate(StatesCache cacheHolder, Layer<?, ?> previous, Updater updater, Optimizer optimizer) {
        int nextLayerSize = nextLayer.getNeurons().size();

        // DOPO
        for (int i = 0; i < neurons.size(); i++) {
            Neuron neuron = neurons.get(i);

            double value = neuron.getValue(cacheHolder);
            double derivative = activation.getFunction().getDerivative(value);

            for (int j = 0; j < nextLayerSize; j++) {
                Synapse synapse = synapses.get(i * nextLayerSize + j);

                float weightChange = calculateGradient(cacheHolder, synapse, derivative);
                updater.acknowledgeChange(synapse, weightChange);
            }
        }
        // PRIMA
//        for (Neuron neuron : neurons) {
//            double value = neuron.getValue(cacheHolder);
//            double derivative = activation.getFunction().getDerivative(value);
//
//            for (Synapse synapse : neuron.getSynapses()) {
//                float weightChange = calculateGradient(cacheHolder, synapse, derivative);
//                updater.acknowledgeChange(synapse, weightChange);
//            }
//        }
    }

    public float calculateGradient(StatesCache cacheHolder, Synapse synapse, double derivative) {
        Neuron input = synapse.getInputNeuron();
        Neuron output = synapse.getOutputNeuron();

        float delta = output.getDelta(cacheHolder);
        float error = clipGradient(synapse.getWeight() * delta * derivative);

        input.setDelta(cacheHolder, error);

        return clipGradient(error * input.getValue(cacheHolder));
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
}
