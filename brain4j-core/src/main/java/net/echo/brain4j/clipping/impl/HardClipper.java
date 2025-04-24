package net.echo.brain4j.clipping.impl;

import net.echo.brain4j.clipping.GradientClipper;
import net.echo.math.tensor.Tensor;

public class HardClipper implements GradientClipper {

    private final double bound;

    public HardClipper(double bound) { this.bound = bound; }

    @Override
    public Tensor clip(Tensor grad) {
        return grad.map(x -> Math.max(-bound, Math.min(bound, x)));
    }
}
