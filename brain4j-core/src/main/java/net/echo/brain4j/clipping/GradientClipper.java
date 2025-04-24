package net.echo.brain4j.clipping;

import net.echo.math.tensor.Tensor;

@FunctionalInterface
public interface GradientClipper {
    Tensor clip(Tensor grad);
}
