package org.brain4j.math.weightsinit.impl;

import org.brain4j.math.weightsinit.WeightInit;

import java.util.random.RandomGenerator;

public class NormalInit implements WeightInit {

    @Override
    public double getBound(int input, int output) {
        return 1;
    }

    @Override
    public double generate(RandomGenerator generator, int input, int output) {
        return randomBetween(generator, -1, 1);
    }
}