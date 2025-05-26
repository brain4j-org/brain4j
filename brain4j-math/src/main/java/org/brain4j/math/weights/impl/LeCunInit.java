package org.brain4j.math.weights.impl;

import org.brain4j.math.weights.WeightInitialization;

import java.util.Random;

public class LeCunInit implements WeightInitialization {

    private static final double SQRT_OF_3 = Math.sqrt(3);

    @Override
    public double getBound(int input, int output) {
        return SQRT_OF_3 / Math.sqrt(input);
    }

    @Override
    public double randomValue(Random generator, int input, int output) {
        double bound = getBound(input, output);
        return randomBetween(generator, -bound, bound);
    }
}