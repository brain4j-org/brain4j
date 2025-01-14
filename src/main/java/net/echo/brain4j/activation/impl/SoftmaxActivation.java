package net.echo.brain4j.activation.impl;

import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.threading.NeuronCacheHolder;

import java.util.List;

public class SoftmaxActivation implements Activation {

    @Override
    public double activate(double input) {
        throw new UnsupportedOperationException("Softmax activation function is not supported for single value");
    }

    @Override
    public double[] activate(double[] inputs) {
        double maxInput = Double.NEGATIVE_INFINITY;

        for (double input : inputs) {
            maxInput = Math.max(maxInput, input);
        }

        double[] expValues = new double[inputs.length];
        double sum = 0.0;

        for (int i = 0; i < inputs.length; i++) {
            sum += (expValues[i] = Math.exp(inputs[i] - maxInput));
        }

        for (int i = 0; i < expValues.length; i++) {
            expValues[i] /= sum;
        }

        return expValues;
    }

    @Override
    public double getDerivative(double input) {
        return input * (1.0 - input);
    }

    @Override
    public void apply(NeuronCacheHolder cacheHolder, List<Neuron> neurons) {
        double[] values = new double[neurons.size()];

        for (int i = 0; i < neurons.size(); i++) {
            values[i] = neurons.get(i).getValue(cacheHolder) + neurons.get(i).getBias();
        }

        double[] activatedValues = activate(values);

        for (int i = 0; i < neurons.size(); i++) {
            neurons.get(i).setValue(cacheHolder, activatedValues[i]);
        }
    }
}
