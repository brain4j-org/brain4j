package net.echo.math4j.math.tensor.autograd;

import net.echo.math4j.math.tensor.Tensor;

public interface Operation {
    Tensor forward(Tensor... inputs);

    Tensor[] backward(Tensor gradOutput, Tensor... inputs);
} 