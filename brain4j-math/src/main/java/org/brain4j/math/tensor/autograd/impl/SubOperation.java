package org.brain4j.math.tensor.autograd.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;

public class SubOperation implements Operation {

    @Override
    public Tensor compute(Tensor... inputs) {
        return inputs[0].minus(inputs[1]);
    }
    
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        return new Tensor[] {
            gradOutput.clone(), 
            gradOutput.times(-1.0f)
        };
    }
} 