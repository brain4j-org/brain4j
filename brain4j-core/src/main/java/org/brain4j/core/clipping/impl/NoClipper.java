package org.brain4j.core.clipping.impl;

import org.brain4j.core.clipping.GradientClipper;
import org.brain4j.math.tensor.Tensor;

public class NoClipper implements GradientClipper {

    @Override
    public void clip(Tensor grad) {
        // Nothing to apply
    }
}
