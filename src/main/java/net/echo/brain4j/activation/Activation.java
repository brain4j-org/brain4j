package net.echo.brain4j.activation;

import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.threading.NeuronCacheHolder;
import net.echo.brain4j.utils.Vector;

import java.util.List;

public interface Activation {

    double activate(double input);

    double[] activate(double[] input);

    double getDerivative(double input);

    default Vector getDerivative(Vector input) {
        Vector result = new Vector(input.size());

        for (int i = 0; i < input.size(); i++) {
            result.set(i, getDerivative(input.get(i)));
        }

        return result;
    }

    void apply(NeuronCacheHolder cacheHolder, List<Neuron> neurons);
}
