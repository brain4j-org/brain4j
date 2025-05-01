package org.brain4j.core.initialization.impl;

import org.brain4j.core.initialization.WeightInitializer;

public class LeCunInit implements WeightInitializer {

    private static final double SQRT_OF_3 = Math.sqrt(3);

    @Override
    public double getBound(int nIn, int nOut) {
        return SQRT_OF_3 / Math.sqrt(nIn);
    }
}
