package org.brain4j.math.weights.impl;

import org.brain4j.math.weights.WeightInitialization;

import java.util.Random;

public class NormalInit implements WeightInitialization {

    @Override
    public double getBound(int input, int output) {
        return 1;
    }

    @Override
    public double randomValue(Random generator, int input, int output) {
        return randomBetween(generator, -1, 1);
    }
}