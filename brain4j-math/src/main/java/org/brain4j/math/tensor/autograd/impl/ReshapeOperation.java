package org.brain4j.math.tensor.autograd.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;

public record ReshapeOperation(int[] newShape) implements Operation {

    @Override
    public int requiredInputs() {
        return 1;
    }

    @Override
    public Tensor compute(Tensor... inputs) {
        return inputs[0].reshape(newShape);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        Tensor input = inputs[0];
        int[] oldShape = input.shape();

        return new Tensor[] { gradOutput.reshape(oldShape) };
    }
}
