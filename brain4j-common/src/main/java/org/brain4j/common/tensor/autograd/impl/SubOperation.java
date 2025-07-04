package org.brain4j.common.tensor.autograd.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.autograd.Operation;

public class SubOperation implements Operation {

    @Override
    public int requiredInputs() {
        return 2;
    }

    @Override
    public Tensor compute(Tensor... inputs) {
        return inputs[0].minus(inputs[1]);
    }
    
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        return new Tensor[] {
            gradOutput.clone(), 
            gradOutput.times(-1.0f)
        };
    }
} 