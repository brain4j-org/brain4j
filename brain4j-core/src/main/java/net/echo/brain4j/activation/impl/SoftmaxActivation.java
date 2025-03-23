package net.echo.brain4j.activation.impl;

import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;
import net.echo.math4j.math.vector.Vector;

import java.util.Arrays;
import java.util.List;

public class SoftmaxActivation implements Activation {

    @Override
    public double activate(double input) {
        throw new UnsupportedOperationException(
                "Softmax is a vector-based activation; use activate(double[]).");
    }

    @Override
    public Tensor activate(Tensor input) {
        return input.softmax();
    }

    @Override
    public double getDerivative(double input) {
        return input * (1.0 - input);
    }

    @Override
    public void apply(StatesCache cacheHolder, List<Neuron> neurons) {
        Tensor vector = TensorFactory.create(neurons.size());

        for (int i = 0; i < neurons.size(); i++) {
            double bias = neurons.get(i).getBias();
            double value = neurons.get(i).getValue(cacheHolder);

            vector.set(value + bias, i);
        }

        Tensor activatedValues = activate(vector);

        for (int i = 0; i < neurons.size(); i++) {
            neurons.get(i).setValue(cacheHolder, activatedValues.get(i));
        }
    }
}
