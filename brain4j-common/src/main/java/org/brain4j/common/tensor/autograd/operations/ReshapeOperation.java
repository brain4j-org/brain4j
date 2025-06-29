package org.brain4j.common.tensor.autograd.operations;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.autograd.Operation;

public class ReshapeOperation implements Operation {

    private final int[] newShape;

    public ReshapeOperation(int[] newShape) {
        this.newShape = newShape;
    }

    @Override
    public Tensor forward(Tensor... inputs) {
        return inputs[0].reshape(newShape);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        Tensor input = inputs[0];
        int[] oldShape = input.shape();

        return new Tensor[] { gradOutput.reshape(oldShape) };
    }
}
