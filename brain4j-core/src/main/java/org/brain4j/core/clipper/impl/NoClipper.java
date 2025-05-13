package org.brain4j.core.clipper.impl;

import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.math.tensor.Tensor;

public class NoClipper implements GradientClipper {

    @Override
    public void clip(Tensor grad) {
        // Nothing to apply
    }
}