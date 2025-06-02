package org.brain4j.core.clipper.impl;

import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.math.tensor.Tensor;

public class HardClipper implements GradientClipper {

    private final double bound;

    public HardClipper(double bound) { this.bound = bound; }

    @Override
    public void clip(Tensor grad) {
        float[] gradData = grad.data();

        for (int i = 0; i < grad.elements(); i++) {
            double clamped = Math.max(-bound, Math.min(bound, gradData[i]));
            gradData[i] = (float) clamped;
        }
    }
}