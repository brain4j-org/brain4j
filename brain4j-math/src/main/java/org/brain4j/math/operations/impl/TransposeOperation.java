package org.brain4j.math.operations.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.operations.Operation;

/**
 * Swaps two tensor axes with autograd. Use it to record transpositions in the
 * computation graph.
 *
 * @param dim1 first axis
 * @param dim2 second axis
 */
public record TransposeOperation(int dim1, int dim2) implements Operation {

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
        return inputs[0].transpose(dim1, dim2);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        return new Tensor[] { gradOutput.transpose(dim1, dim2) };
    }
}
