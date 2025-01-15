package net.echo.brain4j.activation.impl;

import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.threading.NeuronCacheHolder;

import java.util.List;

public class SigmoidActivation implements Activation {

    @Override
    public double activate(double input) {
        return 1 / (1 + Math.exp(-input));
    }

    @Override
    public double[] activate(double[] input) {
        double[] result = new double[input.length];

        for (int i = 0; i < input.length; i++) {
            result[i] = activate(input[i]);
        }

        return result;
    }

    @Override
    public double getDerivative(double input) {
        return input * (1 - input);
    }

    @Override
    public double[][] getDerivativeMatrix(double[] outputs) {
        throw new UnsupportedOperationException("Sigmoid activation function is not supported for multiple inputs");
    }

    @Override
    public void apply(NeuronCacheHolder cacheHolder, List<Neuron> neurons) {
        for (Neuron neuron : neurons) {
            double output = activate(neuron.getValue(cacheHolder) + neuron.getBias());

            neuron.setValue(cacheHolder, output);
        }
    }
}
