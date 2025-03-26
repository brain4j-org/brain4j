package net.echo.brain4j.loss.impl;

import net.echo.brain4j.loss.LossFunction;
import net.echo.math4j.math.tensor.Tensor;

public class CrossEntropy implements LossFunction {

    @Override
    public double calculate(Tensor actual, Tensor predicted) {
        double loss = 0.0;

        for (int i = 0; i < actual.elements(); i++) {
            loss -= actual.get(i) * Math.log(predicted.get(i) + 1e-15);
        }

        return loss;
    }

    @Override
    public double getDelta(double error, double derivative) {
        return error;
    }
}
