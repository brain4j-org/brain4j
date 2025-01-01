package net.echo.brain4j.loss.impl;

import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.utils.Vector;

public class MeanSquaredError implements LossFunction {

    @Override
    public double calculate(Vector actual, Vector predicted) {
        double error = 0.0;

        for (int i = 0; i < actual.size(); i++) {
            error += Math.pow(actual.get(i) - predicted.get(i), 2);
        }

        return error / actual.size();
    }
}
