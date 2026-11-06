package org.brain4j.math.operations;

import org.brain4j.math.tensor.Tensor;

/**
 * Defines a differentiable tensor operation. Use it to implement forward
 * computation plus gradient propagation for autograd.
 */
public interface Operation {

    /**
     * Get the expected input count.
     *
     * @return number of required inputs
     */
    default int requiredInputs() {
        return 2;
    }

    /**
     * Compute the forward result.
     *
     * @param inputs forward inputs
     * @return forward output
     */
    Tensor compute(Tensor... inputs);

    /**
     * Propagate a gradient to the inputs.
     *
     * @param gradOutput upstream gradient
     * @param output forward output
     * @param inputs forward inputs
     * @return one gradient per input
     */
    Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs);
} 