package org.brain4j.core.clipping.impl;

import org.brain4j.core.clipping.GradientClipper;
import org.brain4j.math.tensor.Tensor;

public class HardClipper implements GradientClipper {

    private final double bound;

    public HardClipper(double bound) { this.bound = bound; }

    @Override
    public void clip(Tensor grad) {
        grad.map(x -> Math.max(-bound, Math.min(bound, x)));
    }
}
