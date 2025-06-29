package org.brain4j.core.layer.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;

public class ReshapeLayer extends Layer {

    private final int[] shape;

    public ReshapeLayer(int... shape) {
        this.shape = shape;
    }

    @Override
    public Tensor forward(ForwardContext context) {
        Tensor input = context.input();

        int[] inputShape = input.shape();
        int[] newShape = new int[shape.length + 1];

        newShape[0] = inputShape[0];

        System.arraycopy(shape, 0, newShape, 1, shape.length);

        return input.reshapeGrad(newShape);
    }

    @Override
    public int size() {
        return 0;
    }
}
