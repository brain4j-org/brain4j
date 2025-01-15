package net.echo.brain4j.activation.impl;

import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.threading.NeuronCacheHolder;

import java.util.List;

public class TanhActivation implements Activation {

    @Override
    public double activate(double input) {
        return Math.tanh(input);
    }

    @Override
    public double[] activate(double[] inputs) {
        double[] result = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            result[i] = activate(inputs[i]);
        }
        return result;
    }

    @Override
    public double getDerivative(double input) {
        return 1.0 - Math.pow(Math.tanh(input), 2);
    }

    @Override
    public double[][] getDerivativeMatrix(double[] outputs) {
        throw new UnsupportedOperationException("Tanh activation function is not supported for multiple inputs");
    }

    @Override
    public void apply(NeuronCacheHolder cacheHolder, List<Neuron> neurons) {
        for (Neuron neuron : neurons) {
            double value = neuron.getValue(cacheHolder) + neuron.getBias();
            neuron.setValue(cacheHolder, activate(value));
        }
    }
}
