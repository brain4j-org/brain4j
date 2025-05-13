package org.brain4j.core.weights.impl;

import org.brain4j.core.weights.WeightInitialization;

public class NormalInit implements WeightInitialization {

    @Override
    public double getBound(int input, int output) {
        return 1;
    }
}