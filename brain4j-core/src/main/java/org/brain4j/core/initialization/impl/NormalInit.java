package org.brain4j.core.initialization.impl;

import org.brain4j.core.initialization.WeightInitializer;

public class NormalInit implements WeightInitializer {

    @Override
    public double getBound(int nIn, int nOut) {
        return 1;
    }
}
