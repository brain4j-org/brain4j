package org.brain4j.math.operations.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.operations.Operation;

/**
 * Multiplies two tensors with autograd. Use it to record element-wise products
 * in the computation graph.
 */
public class MulOperation implements Operation {

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor compute(Tensor... inputs) {
        return inputs[0].times(inputs[1]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        Tensor a = inputs[0];
        Tensor b = inputs[1];

        Tensor gradA = gradOutput.times(b);
        Tensor gradB = gradOutput.times(a);

        return new Tensor[] { gradA, gradB };
    }
} 