package org.brain4j.core.weights.impl;

import org.brain4j.core.weights.WeightInitialization;

public class LeCunInit implements WeightInitialization {

    private static final double SQRT_OF_3 = Math.sqrt(3);

    @Override
    public double getBound(int input, int output) {
        return SQRT_OF_3 / Math.sqrt(input);
    }
}