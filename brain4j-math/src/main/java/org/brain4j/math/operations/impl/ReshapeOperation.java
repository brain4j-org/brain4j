package org.brain4j.math.operations.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.operations.Operation;

/**
 * Reshapes a tensor with autograd. Use it to record shape changes in the
 * computation graph.
 *
 * @param newShape target shape
 */
public record ReshapeOperation(int[] newShape) implements Operation {

    /**
     * {@inheritDoc}
     */
    @Override
    public int requiredInputs() {
        return 1;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor compute(Tensor... inputs) {
        return inputs[0].reshape(newShape);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        Tensor input = inputs[0];
        int[] oldShape = input.shape();

        return new Tensor[] { gradOutput.reshape(oldShape) };
    }
}
