package net.echo.brain4j.loss.impl;

import net.echo.brain4j.loss.LossFunction;
import net.echo.math4j.math.vector.Vector;

public class MeanAbsoluteError implements LossFunction {

    @Override
    public double calculate(Vector actual, Vector predicted) {
        double loss = 0.0;

        for (int i = 0; i < actual.size(); i++) {
            loss += Math.abs(actual.get(i) - predicted.get(i));
        }

        return loss / actual.size();
    }

    @Override
    public double getDelta(double error, double derivative) {
        return Math.signum(error);
    }
}
