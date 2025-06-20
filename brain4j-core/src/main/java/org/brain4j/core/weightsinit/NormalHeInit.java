package org.brain4j.core.weightsinit;

import org.brain4j.math.weightsinit.WeightInitialization;

import java.util.Random;

public class NormalHeInit implements WeightInitialization {

    @Override
    public double getBound(int input, int output) {
        return 2.0 / input;
    }

    @Override
    public double generate(Random generator, int input, int output) {
        return randomBetween(generator, 0, getBound(input, output));
    }
}