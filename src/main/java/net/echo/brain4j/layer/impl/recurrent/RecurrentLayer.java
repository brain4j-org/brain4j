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
            throw new UnsupportedOperationException("Previous layer must be a dense or recurrent layer!");
        }

        int prevSize = lastLayer.getNeurons().size();

        Vector hiddenState = previousTimestep.get();

        for (int i = 0; i < neurons.size(); i++) {
            double inputValue = denseLayer.getWeights().get(i).weightedSum(input);
            double recurrentValue = recurrentWeights.get(i).weightedSum(hiddenState);

            double rawState = inputValue + recurrentValue + hiddenStateBias.get(i);
            double newState = activation.getFunction().activate(rawState);

            hiddenState.set(i, newState);
        }

        previousTimestep.set(hiddenState);

        for (int i = 0; i < neurons.size(); i++) {
            double output = outputWeights.get(i).weightedSum(hiddenState);
            neurons.get(i).setValue(cache, output);
        }

        applyFunction(cache, lastLayer);

        Vector outputVector = new Vector(neurons.size());

        for (int i = 0; i < size(); i++) {
            outputVector.set(i, neurons.get(i).getValue(cache));
        }

        return outputVector;
    }

    @Override
    public void propagate(StatesCache cacheHolder, Layer<?, ?> previous, Updater updater, Optimizer optimizer) {
        super.propagate(cacheHolder, previous, updater, optimizer);
    }

    public List<Vector> getRecurrentWeights() {
        return recurrentWeights;
    }
}
