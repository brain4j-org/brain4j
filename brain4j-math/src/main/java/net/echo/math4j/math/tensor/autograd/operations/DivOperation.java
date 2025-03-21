package net.echo.math4j.math.tensor.autograd.operations;

import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.autograd.Operation;

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
        Tensor gradA = gradOutput.times(b.clone().div(1.0));
        
        // d(a/b)/db = -a/b^2
        Tensor gradB = gradOutput.times(a.clone().div(b.clone().times(b.clone())).times(-1.0));
        
        return new Tensor[] { gradA, gradB };
    }
} 