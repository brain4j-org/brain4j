package org.brain4j.common.tensor.autograd.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.autograd.Operation;

public class DivOperation implements Operation {

    @Override
    public Tensor compute(Tensor... inputs) {
        return inputs[0].divide(inputs[1]);
    }
    
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        Tensor a = inputs[0];
        Tensor b = inputs[1];
        
        // d(a/b)/da = 1/b
        Tensor gradA = gradOutput.times(b.divide(1.0f));
        
        // d(a/b)/db = -a/b^2
        Tensor gradB = gradOutput.times(a.divide(b.times(b)).times(-1.0f));
        
        return new Tensor[] { gradA, gradB };
    }
} 