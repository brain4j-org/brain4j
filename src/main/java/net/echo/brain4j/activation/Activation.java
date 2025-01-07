package net.echo.brain4j.activation;

import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.threading.NeuronCacheHolder;

import java.util.List;

public interface Activation {

    double activate(double input);

    double[] activate(double[] input);

    double getDerivative(double input);

    void apply(NeuronCacheHolder cacheHolder, List<Neuron> neurons);
}
