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

import static net.echo.brain4j.utils.MLUtils.clipGradient;

/**
 * Represents a recurrent layer in a neural network.
 */
public class RecurrentLayer extends DenseLayer {

    private List<Vector> recurrentWeights;

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

        List<Neuron> nextNeurons = nextLayer.getNeurons();

        int inSize = neurons.size();
        int outSize = nextNeurons.size();

        Vector currentInput = new Vector(inSize);

        for (int i = 0; i < inSize; i++) {
            currentInput.set(i, neurons.get(i).getValue(cache));
        }

        Vector previousHiddenState = new Vector(inSize);

        for (int i = 0; i < inSize; i++) {
            previousHiddenState.set(i, cache.getHiddenState(neurons.get(i)));
        }

        for (int i = 0; i < outSize; i++) {
            double inputValue = weights.get(i).weightedSum(currentInput);
            double recurrentValue = recurrentWeights.get(i).weightedSum(previousHiddenState);

            double newState = activation.getFunction().activate(inputValue + recurrentValue);

            cache.setHiddenState(nextNeurons.get(i), newState);
            nextNeurons.get(i).setValue(cache, newState);
        }

        nextLayer.applyFunction(cache, this);
        return null;
    }

    @Override
    public void propagate(StatesCache cacheHolder, Layer previous, Updater updater, Optimizer optimizer) {
        for (Neuron neuron : neurons) {
            double value = neuron.getValue(cacheHolder);
            double derivative = activation.getFunction().getDerivative(value);

            double delta = cacheHolder.getDelta(neuron);

            for (int i = 0; i < neurons.size(); i++) {
                Neuron recurrentNeuron = neurons.get(i);

                double recurrentValue = recurrentNeuron.getValue(cacheHolder);
                double recurrentGradient = delta * recurrentValue * derivative;

                double currentRecurrentWeight = recurrentWeights.get(i).get(i);
                double updatedRecurrentWeight = currentRecurrentWeight - optimizer.getLearningRate() * recurrentGradient;

                recurrentWeights.get(i).set(i, updatedRecurrentWeight);
            }

            for (Synapse synapse : neuron.getSynapses()) {
                double weightChange = calculateGradient(cacheHolder, synapse, derivative);
                updater.acknowledgeChange(synapse, weightChange);
            }
        }
    }
}
