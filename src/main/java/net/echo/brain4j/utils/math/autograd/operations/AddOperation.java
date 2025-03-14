package net.echo.brain4j.utils.math.autograd.operations;

import net.echo.brain4j.utils.math.autograd.Operation;
import net.echo.brain4j.utils.math.tensor.Tensor;

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