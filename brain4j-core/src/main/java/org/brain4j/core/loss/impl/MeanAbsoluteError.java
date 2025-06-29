package org.brain4j.core.loss.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.loss.LossFunction;

public class MeanAbsoluteError implements LossFunction {

    @Override
    public double calculate(Tensor actual, Tensor predicted) {
        double loss = 0.0;

        for (int i = 0; i < actual.elements(); i++) {
            loss += Math.abs(actual.get(i) - predicted.get(i));
        }

        return loss / actual.elements();
    }

    @Override
    public Tensor getDelta(Tensor error, Tensor derivative) {
        return error.map(Math::signum);
    }
}