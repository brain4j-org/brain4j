package org.brain4j.core.loss.impl;

import org.brain4j.core.loss.LossFunction;
import org.brain4j.common.tensor.Tensor;

public class CrossEntropy implements LossFunction {

    @Override
    public double calculate(Tensor actual, Tensor predicted) {
        double loss = 0.0;

        for (int i = 0; i < actual.elements(); i++) {
            loss -= actual.get(i) * Math.log(predicted.get(i) + 1e-15);
        }

        return loss / actual.elements();
    }

    @Override
    public Tensor getDelta(Tensor error, Tensor derivative) {
        return error;
    }
}