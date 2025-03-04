package net.echo.brain4j.layer.impl.recurrent;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.convolution.Kernel;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.Synapse;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Represents a recurrent layer in a neural network.
 */
public class RecurrentLayer extends DenseLayer {

    private List<Vector> recurrentWeights;
    private Vector hiddenStateBias;

    /**
     * Constructs an instance of a recurrent layer.
     *
     * @param input the number of neurons in this layer
     * @param activation the activation function to be applied to the output of each neuron
     */
    public RecurrentLayer(int input, Activations activation) {
        super(input, activation);
    }

    @Override
    public void connectAll(Random generator, Layer nextLayer, double bound) {
        super.connectAll(generator, nextLayer, bound);

        int inSize = neurons.size();
        this.recurrentWeights = new ArrayList<>(inSize);
        this.hiddenStateBias = Vector.uniform(-1, 1, inSize);

        for (int i = 0; i < inSize; i++) {
            Vector recurrentWeightsVector = new Vector(inSize);

            for (int j = 0; j < inSize; j++) {
                double value = (generator.nextDouble() * 2 * bound) - bound;
                recurrentWeightsVector.set(j, value);
            }

            recurrentWeights.add(recurrentWeightsVector);
        }
    }

    @Override
    public Kernel forward(StatesCache cache, Layer lastLayer, Kernel input) {
        cache.ensureRecurrentCache();

        if (!(lastLayer instanceof DenseLayer denseLayer)) {
            throw new UnsupportedOperationException("Recurrent layer must be connected to a dense layer or a recurrent layer!");
        }

        List<Neuron> nextNeurons = nextLayer.getNeurons();

        int inSize = neurons.size();
        int outSize = nextNeurons.size();

        Vector currentInput = new Vector(inSize);

        for (int i = 0; i < inSize; i++) {
            currentInput.set(i, neurons.get(i).getValue(cache));
        }

        Vector previousHiddenState = new Vector(inSize);

        for (int i = 0; i < inSize; i++) {
            previousHiddenState.set(i, neurons.get(i).getHiddenState(cache));
        }

        for (int i = 0; i < inSize; i++) {
            Neuron neuron = neurons.get(i);

            double inputValue = denseLayer.getWeights().get(i).weightedSum(currentInput);
            double recurrentValue = recurrentWeights.get(i).weightedSum(previousHiddenState);

            double rawState = inputValue + recurrentValue + hiddenStateBias.get(i);
            double newState = Activations.TANH.getFunction().activate(rawState);

            cache.setHiddenState(neuron, newState);
            // neuron.setValue(cache, newState);
        }

        for (int i = 0; i < outSize; i++) {
            double value = weights.get(i).weightedSum(currentInput);
            nextNeurons.get(i).setValue(cache, value);
        }

        nextLayer.applyFunction(cache, this);
        return null;
    }

    @Override
    public void propagate(StatesCache cacheHolder, Layer previous, Updater updater, Optimizer optimizer) {
        super.propagate(cacheHolder, previous, updater, optimizer);

        for (Neuron neuron : neurons) {
            double value = neuron.getValue(cacheHolder);
            double derivative = activation.getFunction().getDerivative(value);

            double delta = neuron.getDelta(cacheHolder);

            for (Neuron recurrentNeuron : neurons) {
                double recurrentValue = cacheHolder.getHiddenState(recurrentNeuron);
                double recurrentGradient = delta * recurrentValue * derivative;

                updater.acknowledgeRecurrentChange(neuron.getId(), recurrentNeuron.getId(), recurrentGradient);
            }

            for (Synapse synapse : neuron.getSynapses()) {
                double weight = synapse.getWeight();
                double propagatedDelta = delta * weight;

                cacheHolder.addDelta(synapse.getInputNeuron(), propagatedDelta);
            }
        }
    }

    public List<Vector> getRecurrentWeights() {
        return recurrentWeights;
    }
}
