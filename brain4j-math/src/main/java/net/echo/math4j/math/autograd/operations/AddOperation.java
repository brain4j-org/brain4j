package net.echo.math4j.math.autograd.operations;

import net.echo.math4j.math.autograd.Operation;
import net.echo.math4j.math.tensor.Tensor;

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