package org.brain4j.math.operations.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.operations.Operation;

/**
 * Multiplies two matrices with autograd. Use it to record batched products in
 * the computation graph.
 */
public class MatMulOperation implements Operation {

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor compute(Tensor... inputs) {
        return inputs[0].matmul(inputs[1]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        Tensor a = inputs[0];
        Tensor b = inputs[1];

        Tensor aT = a.transpose();
        Tensor bT = b.transpose();

        Tensor gradA = AddOperation.reduceTo(gradOutput.matmul(bT), a);
        Tensor gradB = AddOperation.reduceTo(aT.matmul(gradOutput), b);

        return new Tensor[] { gradA, gradB };
    }
} 