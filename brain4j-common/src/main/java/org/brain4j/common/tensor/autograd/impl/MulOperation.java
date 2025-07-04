package org.brain4j.common.tensor.autograd.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.autograd.Operation;

public class MulOperation implements Operation {

    @Override
    public int requiredInputs() {
        return 2;
    }

    @Override
    public Tensor compute(Tensor... inputs) {
        return inputs[0].times(inputs[1]);
    }
    
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        Tensor a = inputs[0];
        Tensor b = inputs[1];

        Tensor gradA = gradOutput.times(b);  // ∂z/∂a = ∂z/∂out * b
        Tensor gradB = gradOutput.times(a);  // ∂z/∂b = ∂z/∂out * a

        return new Tensor[] { gradA, gradB };
    }
} 