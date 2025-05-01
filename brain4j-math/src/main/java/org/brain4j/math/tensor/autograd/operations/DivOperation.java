package org.brain4j.math.tensor.autograd.operations;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;

public class DivOperation implements Operation {

    @Override
    public Tensor forward(Tensor... inputs) {
        return inputs[0].divide(inputs[1]);
    }
    
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        Tensor a = inputs[0];
        Tensor b = inputs[1];
        
        // d(a/b)/da = 1/b
        Tensor gradA = gradOutput.times(b.divide(1.0));
        
        // d(a/b)/db = -a/b^2
        Tensor gradB = gradOutput.times(a.divide(b.times(b)).times(-1.0));
        
        return new Tensor[] { gradA, gradB };
    }
} 