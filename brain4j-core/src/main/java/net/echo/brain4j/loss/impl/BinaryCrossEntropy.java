package net.echo.brain4j.loss.impl;

import net.echo.brain4j.loss.LossFunction;
import net.echo.math.tensor.Tensor;

public class BinaryCrossEntropy implements LossFunction {

    @Override
    public double calculate(Tensor actual, Tensor predicted) {
        double loss = 0.0;

        for (int i = 0; i < actual.elements(); i++) {
            double p = Math.max(Math.min(predicted.get(i), 1 - 1e-15), 1e-15);
            loss += -actual.get(i) * Math.log(p) - (1 - actual.get(i)) * Math.log(1 - p);
        }

        return loss / actual.elements();
    }

    @Override
    public Tensor getDelta(Tensor error, Tensor derivative) {
        return error;
    }
}
