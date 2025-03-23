package net.echo.brain4j.training.data;

import net.echo.math4j.math.tensor.Tensor;

public record DataRow(Tensor inputs, Tensor outputs) {
}
