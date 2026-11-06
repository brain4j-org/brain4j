package org.brain4j.math.operations.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.operations.Operation;

/**
 * Applies an activation function with autograd. Use it to record element-wise
 * non-linearities in the computation graph.
 *
 * @param activation activation function
 */
public record ActivationOperation(Activation activation) implements Operation {

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
        return activation.activate(inputs[0]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        Tensor gradInput = activation.derivative(inputs[0], output, gradOutput);
        return new Tensor[] { gradInput };
    }
}
