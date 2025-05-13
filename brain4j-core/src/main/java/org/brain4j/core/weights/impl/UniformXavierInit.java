package org.brain4j.core.weights.impl;

import org.brain4j.core.weights.WeightInitialization;

public class UniformXavierInit implements WeightInitialization {

    @Override
    public double getBound(int input, int output) {
        return Math.sqrt(6.0 / (input + output));
    }
}