package net.echo.math.data;

import net.echo.math.tensor.Tensor;

public record Sample(Tensor input, Tensor label) {}
