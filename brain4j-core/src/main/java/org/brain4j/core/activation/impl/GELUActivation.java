package org.brain4j.core.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weightsinit.WeightInitialization;
import org.brain4j.core.weightsinit.NormalXavierInit;

import static org.brain4j.math.constants.Constants.PI;

public class GELUActivation implements Activation {

    @Override
    public WeightInitialization defaultWeightInit() {
        return new NormalXavierInit();
    }

    @Override
    public double activate(double input) {
        return 0.5 * input * (1 + Math.tanh(Math.sqrt(2 / PI) * (input + 0.044715 * Math.pow(input, 3))));
    }

    @Override
    public double derivative(double input) {
        double tanhTerm = Math.tanh(Math.sqrt(2 / PI) * (input + 0.044715 * Math.pow(input, 3)));
        return 0.5 * (1 + tanhTerm) + 0.5 * input * (1 - Math.pow(tanhTerm, 2)) * Math.sqrt(2 / PI) * (1 + 3 * 0.044715 * Math.pow(input, 2));
    }

    @Override
    public String kernelPrefix() {
        return "gelu";
    }
}
