package net.echo.brain4j.utils.math.autograd;

import net.echo.brain4j.utils.math.tensor.Tensor;

public interface Operation {
    Tensor forward(Tensor... inputs);
    Tensor[] backward(Tensor gradOutput, Tensor... inputs);
} 