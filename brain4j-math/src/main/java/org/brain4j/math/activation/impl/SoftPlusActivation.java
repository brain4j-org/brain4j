package org.brain4j.math.activation.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.weightsinit.impl.UniformXavierInit;
import org.brain4j.math.weightsinit.WeightInit;

public class SoftPlusActivation implements Activation {
    
    @Override
    public WeightInit defaultWeightInit() {
        return new UniformXavierInit();
    }

    @Override
    public double activate(double input) {
        if (input > 20) {
            // exp(20) ~ 4.8e8, log(1+exp(20)) ≈ input
            return input;
        } else if (input < -20) {
            // exp(-20) ~ 2e-9, log(1+...) ≈ exp(input)
            return Math.exp(input);
        }
        return Math.log1p(Math.exp(input));
    }

    @Override
    public double derivative(double input) {
        if (input >= 0) {
            double expNeg = Math.exp(-input);
            return 1.0 / (1.0 + expNeg);
        } else {
            double expPos = Math.exp(input);
            return expPos / (1.0 + expPos);
        }
    }

    @Override
    public String getKernelPrefix() {
        return "softplus";
    }

    @Override
    public int getActivationId() {
        return 6;
    }
}
