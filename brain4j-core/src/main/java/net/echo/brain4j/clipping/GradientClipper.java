package net.echo.brain4j.clipping;

import net.echo.math.tensor.Tensor;

@FunctionalInterface
public interface GradientClipper {
    void clip(Tensor grad);
}
