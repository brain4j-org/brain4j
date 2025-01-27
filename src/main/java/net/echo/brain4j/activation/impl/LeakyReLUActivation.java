package net.echo.brain4j.activation.impl;

import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.StatesCache;

import java.util.List;

public class LeakyReLUActivation implements Activation {

    @Override
    public double activate(double input) {
        return Math.max(0.01 * input, input);
    }

    @Override
    public double[] activate(double[] input) {
        throw new UnsupportedOperationException("Leaky ReLU activation function is not supported for multiple inputs");
    }

    @Override
    public double getDerivative(double input) {
        return input > 0 ? 1 : 0.01;
    }

    @Override
    public double[][] getDerivativeMatrix(double[] outputs) {
        throw new UnsupportedOperationException("Leaky ReLU activation function is not supported for multiple inputs");
    }

    @Override
    public void apply(StatesCache cacheHolder, List<Neuron> neurons) {
        for (Neuron neuron : neurons) {
            double output = activate(neuron.getValue(cacheHolder) + neuron.getBias());

            neuron.setValue(cacheHolder, output);
        }
    }
}
