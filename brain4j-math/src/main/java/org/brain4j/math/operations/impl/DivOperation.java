package org.brain4j.math.operations.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.operations.Operation;

/**
 * Divides two tensors with autograd. Use it to record element-wise division
 * in the computation graph.
 */
public class DivOperation implements Operation {

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor compute(Tensor... inputs) {
        return inputs[0].divide(inputs[1]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        Tensor a = inputs[0];
        Tensor b = inputs[1];

        Tensor gradA = gradOutput.divide(b);
        Tensor gradB = gradOutput.times(a.divide(b.times(b)).times(-1.0f));

        return new Tensor[] { gradA, gradB };
    }
} 