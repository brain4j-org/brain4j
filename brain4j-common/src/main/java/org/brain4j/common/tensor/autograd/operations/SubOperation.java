package org.brain4j.common.tensor.autograd.operations;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.autograd.Operation;

public class SubOperation implements Operation {

    @Override
    public Tensor forward(Tensor... inputs) {
        return inputs[0].minus(inputs[1]);
    }
    
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        return new Tensor[] {
            gradOutput.clone(), 
            gradOutput.times(-1.0) 
        };
    }
} 