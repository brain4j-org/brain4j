package org.brain4j.core.training.optimizer.impl;

import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.math.tensor.Tensor;

public class GradientDescent extends Optimizer {

    public GradientDescent(double learningRate) {
        super(learningRate);
    }

    @Override
    public Tensor step(Tensor weights, Tensor gradient) {
        return gradient;
    }
}
