package org.brain4j.math.operations.impl;

import org.brain4j.math.convolution.pooling.impl.MaxPooling;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.operations.Operation;

/**
 * Applies max pooling with autograd. Use it to record spatial downsampling in
 * the computation graph.
 */
public class MaxPoolOperation implements Operation {

    private final MaxPooling pooling;

    /**
     * Create a pooling operation with the given geometry.
     *
     * @param stride step between windows, must be positive
     * @param windowHeight pooling window height, must be positive
     * @param windowWidth pooling window width, must be positive
     */
    public MaxPoolOperation(int stride, int windowHeight, int windowWidth) {
        this.pooling = new MaxPooling(stride, windowHeight, windowWidth);
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
        return pooling.pool(inputs[0]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        return new Tensor[] { pooling.backward(gradOutput, inputs[0]) };
    }
}
