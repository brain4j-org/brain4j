package net.echo.brain4j.loss.impl;

import net.echo.brain4j.loss.LossFunction;
import net.echo.brain4j.utils.Vector;

public class CrossEntropy implements LossFunction {

    @Override
    public double calculate(Vector actual, Vector predicted) {
        double loss = 0.0;

        for (int i = 0; i < actual.size(); i++) {
            loss -= actual.get(i) * Math.log(predicted.get(i) + 1e-12);
        }

        return loss / actual.size();
    }
}
