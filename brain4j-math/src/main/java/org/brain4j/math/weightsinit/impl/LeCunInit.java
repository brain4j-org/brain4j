package org.brain4j.math.weightsinit.impl;

import org.brain4j.math.weightsinit.WeightInit;

import java.util.random.RandomGenerator;

import static org.brain4j.math.Constants.SQRT3;

public class LeCunInit implements WeightInit {

    @Override
    public double getBound(int input, int output) {
        return SQRT3 / Math.sqrt(input);
    }

    @Override
    public double generate(RandomGenerator generator, int input, int output) {
        double bound = getBound(input, output);
        return randomBetween(generator, -bound, bound);
    }
}