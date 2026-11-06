package org.brain4j.math.operations.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.operations.Operation;

/**
 * Subtracts two tensors with autograd. Use it to record element-wise
 * subtraction in the computation graph.
 */
public class SubOperation implements Operation {

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor compute(Tensor... inputs) {
        return inputs[0].minus(inputs[1]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        return new Tensor[] {
            gradOutput.copy(),
            gradOutput.times(-1.0f)
        };
    }
} 