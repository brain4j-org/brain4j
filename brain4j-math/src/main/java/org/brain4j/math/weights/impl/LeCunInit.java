package org.brain4j.math.weights.impl;

import org.brain4j.math.weights.WeightInitialization;

import java.util.Random;

import static org.brain4j.math.constants.Constants.SQRT3;

public class LeCunInit implements WeightInitialization {

    @Override
    public double getBound(int input, int output) {
        return SQRT3 / Math.sqrt(input);
    }

    @Override
    public double randomValue(Random generator, int input, int output) {
        double bound = getBound(input, output);
        return randomBetween(generator, -bound, bound);
    }
}