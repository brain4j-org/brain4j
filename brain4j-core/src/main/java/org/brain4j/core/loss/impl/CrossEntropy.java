package org.brain4j.core.loss.impl;

import org.brain4j.core.loss.LossFunction;
import org.brain4j.math.tensor.Tensor;

public class CrossEntropy implements LossFunction {

    private Tensor classWeights;

    public CrossEntropy() {
    }

    public CrossEntropy(Tensor classWeights) {
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

            loss -= w * y * Math.log(p + 1e-15);
        }

        int batchSize = actual.rank() > 1 ? actual.shapeAt(0) : 1;
        return loss / batchSize;
    }

    @Override
    public Tensor delta(Tensor output, Tensor target, Tensor derivative) {
        Tensor error = output.minus(target);

        return classWeights == null ? error : error.mul(classWeights);
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
