package org.brain4j.core.weightsinit;

import org.brain4j.math.weightsinit.WeightInitialization;

import java.util.Random;

public class NormalInit implements WeightInitialization {

    @Override
    public double getBound(int input, int output) {
        return 1;
    }

    @Override
    public double generate(Random generator, int input, int output) {
        return randomBetween(generator, -1, 1);
    }
}