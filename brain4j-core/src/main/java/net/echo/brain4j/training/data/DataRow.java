package net.echo.brain4j.training.data;

import net.echo.math.tensor.Tensor;

@Deprecated(since = "2.7.0", forRemoval = true)
public record DataRow(Tensor inputs, Tensor outputs) {
}
