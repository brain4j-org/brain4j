package net.echo.math.tensor.autograd;

import net.echo.math.tensor.Tensor;

public interface Operation {

    Tensor forward(Tensor... inputs);

    Tensor[] backward(Tensor gradOutput, Tensor... inputs);
} 