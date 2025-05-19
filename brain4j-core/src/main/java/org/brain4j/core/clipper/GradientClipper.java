package org.brain4j.core.clipper;

import org.brain4j.math.tensor.Tensor;

/**
 * Gradient clipping is a technique used to prevent the gradients
 * from exploding or vanishing during the training process.
 */
@FunctionalInterface
public interface GradientClipper {
    /**
     * Clips the input gradient tensor.
     * @param grad the gradient tensor
     */
    void clip(Tensor grad);
}