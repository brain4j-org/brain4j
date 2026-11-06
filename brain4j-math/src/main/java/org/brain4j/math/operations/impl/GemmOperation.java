package org.brain4j.math.operations.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.operations.Operation;

/**
 * Computes a fused matmul-plus-bias with autograd. Use it to record linear
 * layers with an additive term.
 */
public class GemmOperation implements Operation {

    /**
     * {@inheritDoc}
     */
    @Override
    public int requiredInputs() {
        return 3;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor compute(Tensor... inputs) {
        Tensor input = inputs[0];
        Tensor weight = inputs[1];
        Tensor add = inputs[2];

        if (!checkShape(input, weight)) {
            return compute(input, weight.transpose(), add);
        }

        return input.matmul(weight).plus(add);
    }

    private boolean checkShape(Tensor a, Tensor b) {
        int[] inShape = a.shape();
        int[] weightShape = b.shape();

        int n = inShape[inShape.length - 1];
        int k = weightShape[weightShape.length - 2];

        return n == k;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        Tensor a = inputs[0];
        Tensor b = inputs[1];

        Tensor gradA = AddOperation.reduceTo(gradOutput.matmul(b.transpose()), a);
        Tensor gradB = AddOperation.reduceTo(a.transpose().matmul(gradOutput), b);

        return new Tensor[] { gradA, gradB, gradOutput.copy() };
    }
}
