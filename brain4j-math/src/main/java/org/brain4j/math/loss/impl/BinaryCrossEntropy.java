package org.brain4j.math.loss.impl;

import org.brain4j.math.loss.LossFunction;
import org.brain4j.math.tensor.Tensor;

public class BinaryCrossEntropy implements LossFunction {

    private Tensor classWeights;

    public BinaryCrossEntropy() {
    }

    public BinaryCrossEntropy(Tensor classWeights) {
        this.classWeights = classWeights;
    }

    @Override
    public double calculate(Tensor actual, Tensor predicted) {
        double loss = 0.0;
        int numClasses = actual.shapeAt(actual.rank() - 1);

        for (int i = 0; i < actual.elements(); i++) {
            double y = actual.get(i);
            double p = predicted.get(i);
            double w = 1.0;

            if (classWeights != null) {
                int cls = i % numClasses;
                w = classWeights.get(cls);
            }

            loss -= w * (y * Math.log(p + 1e-15) + (1 - y) * Math.log(1 - p + 1e-15));
        }

        return loss / actual.shapeAt(0);
    }

    @Override
    public Tensor delta(Tensor output, Tensor target, Tensor derivative) {
        Tensor error = output.minus(target);

        if (classWeights == null) return error;

        float w0 = classWeights.get(0);
        float w1 = classWeights.get(1);

        // W = y * w1 + (1 - y) * w0
        Tensor oneMinusTarget = target.mul(-1).plus(1);
        Tensor weightOne = target.times(w1);
        Tensor weightZero = oneMinusTarget.times(w0);
        Tensor W = weightOne.plus(weightZero);

        // delta = error * W
        return error.mul(W);
    }

    @Override
    public boolean isRegression() {
        return false;
    }

    public Tensor getClassWeights() {
        return classWeights;
    }

    public void setClassWeights(Tensor classWeights) {
        this.classWeights = classWeights;
    }
}
