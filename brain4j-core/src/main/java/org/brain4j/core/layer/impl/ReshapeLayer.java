package org.brain4j.core.layer.impl;

import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
import org.brain4j.math.tensor.Tensor;

public class ReshapeLayer extends Layer {

    private final int[] shape;

    public ReshapeLayer(int... shape) {
        this.shape = shape;
    }

    @Override
    public Tensor forward(ForwardContext context) {
        return context.input().reshapeGrad(shape);
    }

    @Override
    public int size() {
        return 0;
    }
}
