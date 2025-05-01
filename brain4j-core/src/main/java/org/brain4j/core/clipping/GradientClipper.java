package org.brain4j.core.clipping;

import org.brain4j.math.tensor.Tensor;

@FunctionalInterface
public interface GradientClipper {
    void clip(Tensor grad);
}
