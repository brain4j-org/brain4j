package org.brain4j.math.tensor.autograd.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;

public class SqueezeOperation implements Operation {

    private int dim = Integer.MAX_VALUE;
    private int[] originalShape;

    public SqueezeOperation() {
    }

    public SqueezeOperation(int dim) {
        this.dim = dim;
    }

    @Override
    public int requiredInputs() {
        return 1;
    }

    @Override
    public Tensor compute(Tensor... inputs) {
        Tensor input = inputs[0].clone();
        this.originalShape = input.shape();
        return dim == Integer.MAX_VALUE ? input.squeeze() : input.squeeze(dim);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        return new Tensor[] { gradOutput.reshape(originalShape) };
    }
}
