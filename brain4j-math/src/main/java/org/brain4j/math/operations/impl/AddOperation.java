package org.brain4j.math.operations.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.operations.Operation;

import java.util.Arrays;

/**
 * Adds two tensors with broadcasting and autograd. Use it to record
 * element-wise addition in the computation graph.
 */
public class AddOperation implements Operation {

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor compute(Tensor... inputs) {
        return inputs[0].plus(inputs[1]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        return new Tensor[] { reduceTo(gradOutput, inputs[0]), reduceTo(gradOutput, inputs[1]) };
    }

    /**
     * Reduce a gradient to a broadcasted shape.
     *
     * @param grad upstream gradient
     * @param like tensor defining the target shape
     * @return gradient reduced to the target shape
     */
    static Tensor reduceTo(Tensor grad, Tensor like) {
        if (grad.rank() == like.rank() && Arrays.equals(grad.shape(), like.shape())) {
            return grad.copy();
        }

        Tensor reduced = grad;

        while (reduced.rank() > like.rank()) {
            reduced = reduced.sum(0, false);
        }

        for (int dim = 0; dim < like.rank(); dim++) {
            if (like.shape()[dim] == 1 && reduced.shape()[dim] != 1) {
                reduced = reduced.sum(dim, true);
            }

        }

        return reduced.reshape(like.shape());
    }
}
