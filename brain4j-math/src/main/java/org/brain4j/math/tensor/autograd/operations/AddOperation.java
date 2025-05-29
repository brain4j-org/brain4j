package org.brain4j.math.tensor.autograd.operations;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;

public class AddOperation implements Operation {

    @Override
    public Tensor forward(Tensor... inputs) {
        return inputs[0].plus(inputs[1]);
    }
    
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        return new Tensor[] { gradOutput.clone(), gradOutput.clone() };
    }
} 