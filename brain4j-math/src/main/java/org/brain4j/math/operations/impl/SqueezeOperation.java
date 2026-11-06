package org.brain4j.math.operations.impl;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.operations.Operation;

/**
 * Removes size-one dimensions with autograd. Use it to record squeezing in the
 * computation graph.
 */
public class SqueezeOperation implements Operation {

    private int dim = Integer.MAX_VALUE;
    private int[] originalShape;

    /**
     * Create an operation squeezing all size-one dimensions.
     */
    public SqueezeOperation() {
    }

    /**
     * Create an operation squeezing one dimension.
     *
     * @param dim axis to squeeze
     */
    public SqueezeOperation(int dim) {
        this.dim = dim;
    }

    /**
     * Get the squeezed dimension.
     *
     * @return dimension, or max value when squeezing all axes
     */
    public int dim() {
        return dim;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int requiredInputs() {
        return 1;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor compute(Tensor... inputs) {
        Tensor input = inputs[0].copy();
        this.originalShape = input.shape();
        return dim == Integer.MAX_VALUE ? input.squeeze() : input.squeeze(dim);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        return new Tensor[] { gradOutput.reshape(originalShape) };
    }
}
