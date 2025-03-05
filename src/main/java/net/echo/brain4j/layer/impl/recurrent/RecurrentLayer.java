package net.echo.brain4j.layer.impl.recurrent;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.structure.Neuron;
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
    private ThreadLocal<Vector> previousTimestep;

    private List<Vector> outputWeights;
    private Vector outputBias;

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
    public void connectAll(Random generator, Layer<?, ?> nextLayer, double bound) {
        super.connectAll(generator, nextLayer, bound);

        int inSize = neurons.size();

        this.previousTimestep = ThreadLocal.withInitial(() -> new Vector(inSize));
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

        this.outputWeights = new ArrayList<>(inSize);
        this.outputBias = Vector.uniform(-1, 1, inSize);

        for (int i = 0; i < inSize; i++) {
            Vector outputWeightVector = new Vector(inSize);

            for (int j = 0; j < inSize; j++) {
                double value = (generator.nextDouble() * 2 * bound) - bound;
                outputWeightVector.set(j, value);
            }

            outputWeights.add(outputWeightVector);
        }
    }

    @Override
    public Vector forward(StatesCache cache, Layer<?, ?> lastLayer, Vector input) {
        if (!(lastLayer instanceof DenseLayer denseLayer)) {
            throw new UnsupportedOperationException("Recurrent layer must be connected to a dense layer or a recurrent layer!");
        }

        List<Neuron> prevNeurons = lastLayer.getNeurons();

        int prevSize = prevNeurons.size();

        Vector currentInput = new Vector(prevSize);

        for (int i = 0; i < prevSize; i++) {
            currentInput.set(i, lastLayer.getNeuronAt(i).getValue(cache));
        }

        Vector timestep = previousTimestep.get();

        for (int i = 0; i < neurons.size(); i++) {
            double inputValue = denseLayer.getWeights().get(i).weightedSum(currentInput);
            double recurrentValue = recurrentWeights.get(i).weightedSum(timestep);

            double rawState = inputValue + recurrentValue + hiddenStateBias.get(i);
            double newState = Activations.TANH.getFunction().activate(rawState);

            timestep.set(i, newState);
            neurons.get(i).setValue(cache, newState);
        }

        previousTimestep.set(timestep);
        applyFunction(cache, lastLayer);

        return null;
    }

    @Override
    public void propagate(StatesCache cacheHolder, Layer<?, ?> previous, Updater updater, Optimizer optimizer) {
        super.propagate(cacheHolder, previous, updater, optimizer);
    }

    public List<Vector> getRecurrentWeights() {
        return recurrentWeights;
    }
}
