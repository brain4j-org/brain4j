package org.brain4j.math.tensor.autograd.impl;

import org.brain4j.math.gpu.ops.FlashAttention;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;

/**
 * This operation wraps the FlashAttention forward and backward passes,
 * enabling automatic differentiation through the attention mechanism.
 * <p>
 * The operation stores the Log-Sum-Exp (LSE) values computed during forward
 * for use in the backward pass, following the FlashAttention paper's approach
 * of recomputing attention scores rather than storing the full attention matrix.
 *
 * @author Adversing
 */
public class FlashAttentionOperation implements Operation {

    private final double scale;
    private final boolean causal;

    // cached LSE and output for backward pass
    private Tensor lse;
    private Tensor output;

    /**
     * Creates a new FlashAttention operation.
     *
     * @param scale  the scaling factor for attention scores (typically 1/sqrt(head_dim))
     * @param causal if true, applies causal masking for autoregressive attention
     */
    public FlashAttentionOperation(double scale, boolean causal) {
        this.scale = scale;
        this.causal = causal;
    }

    @Override
    public int requiredInputs() {
        return 3; // Q, K, V
    }

    @Override
    public Tensor compute(Tensor... inputs) {
        Tensor q = inputs[0];
        Tensor k = inputs[1];
        Tensor v = inputs[2];

        // let's use forward with LSE for training (needed for backward)
        if (q.usesGrad() || k.usesGrad() || v.usesGrad()) {
            Tensor[] result = FlashAttention.forwardWithLse(q, k, v, scale, causal);
            if (result != null) {
                this.output = result[0];
                this.lse = result[1];
                return output;
            }
        }

        Tensor result = FlashAttention.forward(q, k, v, scale, causal);
        if (result != null) {
            this.output = result;
            return result;
        }

        throw new IllegalStateException("FlashAttention requires GpuTensor inputs");
    }

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        Tensor q = inputs[0];
        Tensor k = inputs[1];
        Tensor v = inputs[2];

        if (lse == null || this.output == null) {
            throw new IllegalStateException("Forward pass must be called before backward pass");
        }

        Tensor[] grads = FlashAttention.backward(q, k, v, this.output, gradOutput, lse, scale, causal);

        if (grads == null) {
            throw new IllegalStateException("FlashAttention backward failed - inputs must be GpuTensors");
        }

        // [Q, K, V]
        return grads;
    }

    /**
     * Returns the cached Log-Sum-Exp values from the forward pass.
     *
     * @return the LSE tensor, or null if forward has not been called
     */
    public Tensor getLse() {
        return lse;
    }

    /**
     * Returns the cached output from the forward pass.
     *
     * @return the output tensor, or null if forward has not been called
     */
    public Tensor getOutput() {
        return output;
    }
}

