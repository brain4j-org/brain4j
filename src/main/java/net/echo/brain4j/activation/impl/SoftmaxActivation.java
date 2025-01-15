package net.echo.brain4j.activation.impl;

import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.threading.NeuronCacheHolder;

import java.util.Arrays;
import java.util.List;

public class SoftmaxActivation implements Activation {

    @Override
    public double activate(double input) {
        throw new UnsupportedOperationException(
                "Softmax is a vector-based activation; use activate(double[]).");
    }

    @Override
    public double[] activate(double[] inputs) {
        double maxInput = Arrays.stream(inputs).max().orElse(0.0);

        double[] expValues = new double[inputs.length];
        double sum = 0.0;

        for (int i = 0; i < inputs.length; i++) {
            expValues[i] = Math.exp(inputs[i] - maxInput);
            sum += expValues[i];
        }

        for (int i = 0; i < expValues.length; i++) {
            expValues[i] /= sum;
        }

        return expValues;
    }

    @Override
    public double getDerivative(double input) {
        throw new UnsupportedOperationException(
                "Softmax derivative is multi-dimensional; use getDerivativeMatrix(double[]).");
    }

    @Override
    public double[][] getDerivativeMatrix(double[] outputs) {
        int n = outputs.length;
        double[][] jacobian = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    jacobian[i][j] = outputs[i] * (1.0 - outputs[i]);
                } else {
                    jacobian[i][j] = -outputs[i] * outputs[j];
                }
            }
        }
        return jacobian;
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
