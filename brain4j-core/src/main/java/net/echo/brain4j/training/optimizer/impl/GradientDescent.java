package net.echo.brain4j.training.optimizer.impl;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.training.optimizer.Optimizer;
import net.echo.math.tensor.Tensor;

public class GradientDescent extends Optimizer {

    public GradientDescent(double learningRate) {
        super(learningRate);
    }

    @Override
    public Tensor optimize(Layer layer, Tensor delta, Tensor output) {
        return delta.matmul(output);
    }
}
