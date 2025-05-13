package org.brain4j.core.loss;

import org.brain4j.core.loss.impl.BinaryCrossEntropy;
import org.brain4j.core.loss.impl.CrossEntropy;
import org.brain4j.core.loss.impl.MeanSquaredError;

public enum Loss {

    BINARY_CROSS_ENTROPY(new BinaryCrossEntropy()),
    CROSS_ENTROPY(new CrossEntropy()),
    MEAN_ABSOLUTE_ERROR(new MeanSquaredError()),
    MEAN_SQUARED_ERROR(new MeanSquaredError());

    private final LossFunction function;

    Loss(LossFunction function) {
        this.function = function;
    }

    public LossFunction function() {
        return function;
    }
}
