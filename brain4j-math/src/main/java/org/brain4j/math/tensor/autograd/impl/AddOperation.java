package org.brain4j.math.tensor.autograd.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;

import java.util.Arrays;

public class AddOperation implements Operation {

    @Override
    public Tensor compute(Tensor... inputs) {
        return inputs[0].plus(inputs[1]);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        return new Tensor[] { reduceTo(gradOutput, inputs[0]), reduceTo(gradOutput, inputs[1]) };
    }

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
