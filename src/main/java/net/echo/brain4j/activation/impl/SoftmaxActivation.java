package net.echo.brain4j.activation.impl;

import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.StatesCache;
import net.echo.brain4j.utils.Vector;

import java.util.Arrays;
import java.util.List;

public class SoftmaxActivation implements Activation {

    @Override
    public double activate(double input) {
        throw new UnsupportedOperationException(
                "Softmax is a vector-based activation; use activate(double[]).");
    }

    @Override
    public Vector activate(Vector input) {
        double maxInput = Arrays.stream(input.toArray()).max().orElse(0.0);

        Vector expValues = new Vector(input.size());
        double sum = 0.0;

        for (int i = 0; i < input.size(); i++) {
            expValues.set(i, Math.exp(input.get(i) - maxInput));
            sum += expValues.get(i);
        }

        for (int i = 0; i < expValues.size(); i++) {
            expValues.set(i, expValues.get(i) / sum);
        }

        return expValues;
    }

    @Override
    public double getDerivative(double input) {
        return input * (1.0 - input);
    }

    @Override
    public void apply(StatesCache cacheHolder, List<Neuron> neurons) {
        Vector vector = new Vector(neurons.size());

        for (int i = 0; i < neurons.size(); i++) {
            vector.set(i, neurons.get(i).getValue(cacheHolder) + neurons.get(i).getBias());
        }

        Vector activatedValues = activate(vector);

        for (int i = 0; i < neurons.size(); i++) {
            neurons.get(i).setValue(cacheHolder, activatedValues.get(i));
        }
    }
}
