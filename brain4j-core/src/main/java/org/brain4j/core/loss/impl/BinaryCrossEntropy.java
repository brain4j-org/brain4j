package org.brain4j.core.loss.impl;

import org.brain4j.common.Commons;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.loss.LossFunction;

public class BinaryCrossEntropy implements LossFunction {

    @Override
    public double calculate(Tensor expected, Tensor predicted) {
        double loss = 0.0;

        for (int i = 0; i < expected.elements(); i++) {
            double p = Commons.clamp(predicted.get(i), 1e-15, 1 - 1e-15);
            loss += -expected.get(i) * Math.log(p) - (1 - expected.get(i)) * Math.log(1 - p);
        }

        return loss / expected.elements();
    }

    @Override
    public Tensor getDelta(Tensor error, Tensor derivative) {
        return error;
    }
}
