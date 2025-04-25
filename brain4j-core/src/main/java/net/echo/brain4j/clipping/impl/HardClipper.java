package net.echo.brain4j.clipping.impl;

import net.echo.brain4j.clipping.GradientClipper;
import net.echo.math.tensor.Tensor;

public class HardClipper implements GradientClipper {

    private final double bound;

    public HardClipper(double bound) { this.bound = bound; }

    @Override
    public void clip(Tensor grad) {
        grad.map(x -> Math.max(-bound, Math.min(bound, x)));
    }
}
