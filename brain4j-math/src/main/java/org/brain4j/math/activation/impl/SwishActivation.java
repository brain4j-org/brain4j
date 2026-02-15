package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weightsinit.impl.NormalXavierInit;
import org.brain4j.math.weightsinit.WeightInit;

public class SwishActivation implements Activation {

    @Override
    public WeightInit defaultWeightInit() {
        return new NormalXavierInit();
    }

    @Override
    public double activate(double input) {
        return input * (1.0 / (1.0 + Math.exp(-input)));
    }

    @Override
    public double derivative(double input) {
        double sigmoid = 1.0 / (1.0 + Math.exp(-input));
        return sigmoid + input * sigmoid * (1 - sigmoid);
    }

    @Override
    public String getKernelPrefix() {
        return "swish";
    }

    @Override
    public int getActivationId() {
        return 5;
    }
}
