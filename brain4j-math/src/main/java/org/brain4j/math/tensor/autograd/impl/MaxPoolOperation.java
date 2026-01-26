package org.brain4j.math.tensor.autograd.impl;

import org.brain4j.math.convolution.pooling.impl.MaxPooling;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;

public class MaxPoolOperation implements Operation {

    private final MaxPooling pooling;

    public MaxPoolOperation(int stride, int windowHeight, int windowWidth) {
        this.pooling = new MaxPooling(stride, windowHeight, windowWidth);
    }

    @Override
    public int requiredInputs() {
        return 1;
    }

    @Override
    public Tensor compute(Tensor... inputs) {
        return pooling.pool(inputs[0]);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        return new Tensor[] { pooling.backward(gradOutput, inputs[0]) };
    }
}
