package org.brain4j.math.operations;

import org.brain4j.math.tensor.Tensor;

/**
 * Tracks gradients and graph links for one tensor. Use it to accumulate
 * gradients and propagate them through the recorded operation.
 */
public class AutogradContext {

    private final Tensor owner;
    private final boolean requiresGrad;
    private Tensor[] inputs;
    private Tensor grad;
    private Operation operation;
    private int expectedContributions = 0;
    private int receivedContributions = 0;

    /**
     * Create a context for a tensor.
     *
     * @param owner tensor owning this context
     * @param requiresGrad whether the owner tracks gradients
     */
    public AutogradContext(Tensor owner, boolean requiresGrad) {
        this.owner = owner;
        this.requiresGrad = requiresGrad;
        this.grad = null;
    }

    /**
     * Clear the accumulated gradient and counters.
     */
    public void zeroGrad() {
        this.grad = null;
        this.receivedContributions = 0;
        this.expectedContributions = 0;
    }

    /**
     * Record the operation that produced the owner.
     *
     * @param operation forward operation
     * @param inputs forward inputs
     */
    public void setOperation(Operation operation, Tensor... inputs) {
        this.operation = operation;
        this.inputs = inputs;

        if (inputs == null) {
            return;
        }

        for (Tensor input : inputs) {
            if (input == null || !input.usesGrad()) {
                continue;
            }

            input.autogradContext().increaseContributes();
        }

    }

    /**
     * Count one more expected gradient contribution.
     */
    public void increaseContributes() {
        expectedContributions++;
    }

    /**
     * Check whether the owner tracks gradients.
     *
     * @return true when gradients are required
     */
    public boolean requiresGrad() {
        return requiresGrad;
    }

    /**
     * Get the accumulated gradient.
     *
     * @return gradient, or null if absent
     */
    public Tensor grad() {
        return grad;
    }

    /**
     * Accumulate a gradient and propagate once complete.
     *
     * @param gradOutput upstream gradient
     */
    public void backward(Tensor gradOutput) {
        if (!requiresGrad) {
            return;
        }

        this.grad = grad == null ? gradOutput.copy() : grad.add(gradOutput.broadcastLike(grad));

        receivedContributions++;

        int needed = Math.max(1, expectedContributions);

        if (receivedContributions < needed) {
            return;
        }

        if (operation == null) {
            return;
        }

        Tensor[] inputGrads = operation.backward(grad, owner, inputs);

        for (int i = 0; i < inputs.length; i++) {
            Tensor input = inputs[i];

            if (input == null || !input.usesGrad()) {
                continue;
            }

            input.backward(inputGrads[i]);
        }

    }

    /**
     * Get forward inputs.
     *
     * @return forward inputs, or null if unset
     */
    public Tensor[] inputs() {
        return inputs;
    }

    /**
     * Get the recorded operation.
     *
     * @return operation, or null if the tensor is a leaf
     */
    public Operation operation() {
        return operation;
    }
}