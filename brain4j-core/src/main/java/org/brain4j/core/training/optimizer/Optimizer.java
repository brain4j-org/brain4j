package org.brain4j.core.training.optimizer;

import org.brain4j.core.training.optimizer.impl.Adam;
import org.brain4j.core.training.optimizer.impl.AdamW;
import org.brain4j.core.training.optimizer.impl.GradientDescent;
import org.brain4j.core.training.optimizer.impl.Lion;
import org.brain4j.math.tensor.Tensor;

/**
 * Base class for all gradient-based optimization algorithms.
 * <p>
 * An {@code Optimizer} defines how model parameters (weights) are updated
 * given their gradients after backpropagation. Concrete implementations
 * such as {@link GradientDescent}, {@link Adam}, {@link AdamW}, or {@link Lion}
 * implement different strategies to improve convergence speed, stability,
 * or generalization.
 * </p>
 * @see GradientDescent
 * @see Adam
 * @see AdamW
 * @see Lion
 */
public abstract class Optimizer {

    private double learningRate;
    
    /**
     * Creates a new optimizer with the given learning rate.
     * @param learningRate the step size used to scale gradient updates
     */
    public Optimizer(double learningRate) {
        this.learningRate = learningRate;
    }
    
    /**
     * Performs a single optimization step.
     * <p>
     * This method updates the provided weights based on the gradient tensor
     * and the specific optimization algorithm implemented by the subclass.
     * Note: this method delegates to {@link #step(Tensor, Tensor)} with the pre-calculated gradient.
     *
     * @param weights the weights tensor
     * @return the optimized weight tensor
     */
    public Tensor step(Tensor weights) {
        return step(weights, weights.grad());
    }
    
    /**
     * Performs a single optimization step.
     * <p>
     * This method updates the provided weights based on the gradient tensor
     * and the specific optimization algorithm implemented by the subclass.
     *
     * @param weights the weights tensor
     * @param gradient the gradient of the loss with respect to the weights
     * @return the optimized weight tensor
     */
    public abstract Tensor step(Tensor weights, Tensor gradient);
    
    /**
     * Optional initialization hook for optimizers that require internal state.
     * <p>Default implementation does nothing.
     */
    public void initialize() {
        // Optional hook
    }
    
    /**
     * Optional hook executed after each training batch.
     * <p>Default implementation does nothing.
     */
    public void postBatch() {
        // Optional hook
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }
}