package org.brain4j.math.tensor.autograd.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;

public record TransposeOperation(int dim1, int dim2) implements Operation {

    @Override
    public int requiredInputs() {
        return 1;
    }

    @Override
    public Tensor compute(Tensor... inputs) {
        return inputs[0].transpose(dim1, dim2);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        return new Tensor[] { gradOutput.transpose(dim1, dim2) };
    }
}
