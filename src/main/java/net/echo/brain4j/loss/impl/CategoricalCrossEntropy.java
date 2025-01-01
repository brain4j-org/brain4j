package net.echo.brain4j.loss.impl;

import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.utils.Vector;

public class CategoricalCrossEntropy implements LossFunction {

    @Override
    public double calculate(Vector expected, Vector actual) {
        double sum = 0.0;

        for (int i = 0; i < expected.size(); i++) {
            sum += -expected.get(i) * Math.log(actual.get(i) + 1e-15);
        }

        return sum;
    }
}
