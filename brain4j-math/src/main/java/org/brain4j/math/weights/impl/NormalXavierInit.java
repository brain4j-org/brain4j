package org.brain4j.math.weights.impl;

import org.brain4j.math.weights.WeightInitialization;

import java.util.Random;

public class NormalXavierInit implements WeightInitialization {

    @Override
    public double getBound(int input, int output) {
        return 2.0 / (input + output);
    }

    @Override
    public double generate(Random generator, int input, int output) {
        return randomBetween(generator, 0, getBound(input, output));
    }
}