package org.brain4j.common.tensor.autograd.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.autograd.Operation;

public class AddOperation implements Operation {

    @Override
    public Tensor compute(Tensor... inputs) {
        return inputs[0].plus(inputs[1]);
    }
    
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        return new Tensor[] { gradOutput.clone(), gradOutput.clone() };
    }
} 