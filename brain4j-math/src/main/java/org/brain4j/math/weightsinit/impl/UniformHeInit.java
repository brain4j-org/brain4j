package org.brain4j.math.weightsinit.impl;

import java.util.random.RandomGenerator;

import org.brain4j.math.weightsinit.WeightInit;

/**
 * He uniform init. Use it for ReLU layers needing uniform variance scaling.
 */
public class UniformHeInit implements WeightInit {

    /**
     * {@inheritDoc}
     */
    @Override
    public double bound(int input, int output) {
        return Math.sqrt(6.0 / input);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double generate(RandomGenerator generator, int input, int output) {
        double bound = bound(input, output);
        return randomBetween(generator, -bound, bound);
    }
}
