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
        if (!(lastLayer instanceof DenseLayer denseLayer)) {
            throw new UnsupportedOperationException("Recurrent layer must be connected to a dense layer or a recurrent layer!");
        }

        cache.ensureRecurrentCache();

        List<Neuron> prevNeurons = lastLayer.getNeurons();

        int prevSize = prevNeurons.size();
        int inSize = neurons.size();

        Vector currentInput = new Vector(prevSize);

        for (int i = 0; i < prevSize; i++) {
            currentInput.set(i, lastLayer.getNeuronAt(i).getValue(cache));
        }

        // TODO: Found the issue!
        // We are calculating the hidden state of the current layer and the output of the next layer.
        // How to solve? Make the current layer calculate the values of this layer
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
            neuron.setValue(cache, newState);
        }

        applyFunction(cache, lastLayer);
        return null;
    }

    @Override
    public void propagate(StatesCache cacheHolder, Layer previous, Updater updater, Optimizer optimizer) {
        super.propagate(cacheHolder, previous, updater, optimizer);

        int n = neurons.size();

        for (int i = 0; i < n; i++) {
            Neuron currentNeuron = neurons.get(i);
            double output = currentNeuron.getValue(cacheHolder);

            double derivative = activation.getFunction().getDerivative(output);
            double delta = currentNeuron.getDelta(cacheHolder);

            double errorSignal = clipGradient(delta * derivative);

            for (int j = 0; j < n; j++) {
                Neuron recurrentNeuron = neurons.get(j);
                double previousHidden = cacheHolder.getHiddenState(recurrentNeuron);

                double recurrentGradient = clipGradient(errorSignal * previousHidden);
                updater.acknowledgeRecurrentChange(currentNeuron.getId(), recurrentNeuron.getId(), recurrentGradient);

                double recurrentWeight = recurrentWeights.get(i).get(j);
                double recurrentError = clipGradient(errorSignal * recurrentWeight);

                cacheHolder.addDelta(recurrentNeuron, recurrentError);
            }
        }
    }


    public List<Vector> getRecurrentWeights() {
        return recurrentWeights;
    }
}
