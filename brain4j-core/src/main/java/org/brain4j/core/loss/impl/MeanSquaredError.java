package org.brain4j.core.loss.impl;

import org.brain4j.core.loss.LossFunction;
import org.brain4j.math.tensor.Tensor;

public class MeanSquaredError implements LossFunction {

    @Override
    public double calculate(Tensor actual, Tensor predicted) {
        double loss = 0.0;

        for (int i = 0; i < actual.elements(); i++) {
            loss += Math.pow(actual.get(i) - predicted.get(i), 2);
        }

        return loss / actual.elements();
    }

    @Override
    public Tensor getDelta(Tensor error, Tensor derivative) {
        return error.mul(derivative);
    }
}