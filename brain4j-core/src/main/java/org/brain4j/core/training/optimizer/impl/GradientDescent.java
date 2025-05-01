package org.brain4j.core.training.optimizer.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.math.tensor.Tensor;

public class GradientDescent extends Optimizer {

    public GradientDescent(double learningRate) {
        super(learningRate);
    }

    @Override
    public Tensor optimize(Layer layer, Tensor delta, Tensor output) {
        return output.transpose().matmul(delta);
    }
}
