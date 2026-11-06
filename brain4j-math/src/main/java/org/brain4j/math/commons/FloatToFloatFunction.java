package org.brain4j.math.commons;

/**
 * Single-input scalar function. Use it for activations, solvers, or plotting.
 */
@FunctionalInterface
public interface FloatToFloatFunction {
    /**
     * Applies the function to the given input.
     *
     * @return the function value
     */
    float apply(float input);
}
