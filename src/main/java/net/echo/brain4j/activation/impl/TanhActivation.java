package net.echo.brain4j.activation.impl;

import net.echo.brain4j.activation.Activation;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.StatesCache;

import java.util.List;

public class TanhActivation implements Activation {

    @Override
    public double activate(double input) {
        return Math.tanh(input);
    }

    @Override
    public double getDerivative(double input) {
        return 1.0 - Math.pow(Math.tanh(input), 2);
    }
}
