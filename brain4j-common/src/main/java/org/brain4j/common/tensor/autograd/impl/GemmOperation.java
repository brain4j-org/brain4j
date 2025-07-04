package org.brain4j.common.tensor.autograd.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.autograd.Operation;

public class GemmOperation implements Operation {

    @Override
    public int requiredInputs() {
        return 3;
    }

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

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        Tensor a = inputs[0];
        Tensor b = inputs[1];

        // For matrix multiplication: C = A @ B
        // dL/dA = dL/dC @ B.T
        Tensor gradA = gradOutput.matmul(b.transpose());

        // dL/dB = A.T @ dL/dC
        Tensor gradB = a.transpose().matmul(gradOutput);

        return new Tensor[] { gradA, gradB, gradOutput.clone() };
    }
}
