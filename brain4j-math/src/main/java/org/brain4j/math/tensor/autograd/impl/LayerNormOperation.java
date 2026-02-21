package org.brain4j.math.tensor.autograd.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;

public record LayerNormOperation(double epsilon) implements Operation {

    @Override
    public int requiredInputs() {
        return 3;
    }

    @Override
    public Tensor compute(Tensor... inputs) {
        return inputs[0].layerNorm(epsilon).mul(inputs[1]).add(inputs[2]);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        Tensor a = inputs[0];
        Tensor b = inputs[1];

        Tensor gradA = gradOutput.times(b);  // ∂z/∂a = ∂z/∂out * b
        Tensor gradB = gradOutput.times(a);  // ∂z/∂b = ∂z/∂out * a

        return new Tensor[] { gradA, gradB, gradOutput.copy() };
    }
}
