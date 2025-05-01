package org.brain4j.core.initialization.impl;

import org.brain4j.core.initialization.WeightInitializer;

public class HeInit implements WeightInitializer {

    @Override
    public double getBound(int nIn, int nOut) {
        return Math.sqrt(2.0 / nIn);
    }
}
