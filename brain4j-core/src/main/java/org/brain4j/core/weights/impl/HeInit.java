package org.brain4j.core.weights.impl;

import org.brain4j.core.weights.WeightInitialization;

public class HeInit implements WeightInitialization {

    @Override
    public double getBound(int input, int output) {
        return Math.sqrt(2.0 / input);
    }
}