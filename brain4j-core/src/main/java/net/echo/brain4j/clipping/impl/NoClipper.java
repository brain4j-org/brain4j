package net.echo.brain4j.clipping.impl;

import net.echo.brain4j.clipping.GradientClipper;
import net.echo.math.tensor.Tensor;

public class NoClipper implements GradientClipper {

    @Override
    public void clip(Tensor grad) {
        // Nothing to apply
    }
}
