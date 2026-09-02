package org.brain4j.core.training.updater;

import org.brain4j.core.training.updater.impl.NormalUpdater;
import org.brain4j.core.training.updater.impl.StochasticUpdater;
import org.brain4j.math.tensor.Tensor;

import java.util.HashMap;
import java.util.Map;

/**
 * Abstract base class for implementing gradient-based weight updaters.
 * <p>
 * This class provides a common interface and core logic for accumulating gradients 
 * and applying weight updates to a model.
 * <p>
 * Subclasses should override {@link #postBatch} or {@link #postFit} as needed to update the weights.
 *
 * @see StochasticUpdater
 * @see NormalUpdater
 */
public abstract class Updater {

    protected final Map<Tensor, Tensor> weightsGradients = new HashMap<>();
    
    /**
     * Applies the accumulated gradients to the model's weights.
     * <p>
     * Each weight tensor is updated in-place using the formula:
     * <pre>
     * {@code weights -= (gradient / samples) * learningRate}
     * </pre>
     *
     * @param samples the number of samples used to normalize the gradient
     */
    public void updateWeights(double learningRate, int samples) {
        for (Map.Entry<Tensor, Tensor> entry : weightsGradients.entrySet()) {
            Tensor weights = entry.getKey();
            Tensor gradient = entry.getValue();
            
            if (gradient == null || weights == null) continue;

            while (gradient.rank() > weights.rank()) {
                gradient = gradient.sum(0, false);
            }

            weights.sub(gradient.div(samples).mul(learningRate));
        }
    }

    /**
     * Schedules a gradient update for the given weight tensor.
     * <p>
     * If a gradient is already scheduled for the specified weight tensor,
     * the new gradient is added to the existing one.
     * This allows accumulation of gradients across mini-batches or iterations.
     *
     * @param weights the weight tensor to be updated
     * @param gradient the gradient corresponding to the weight tensor
     */
    public void change(Tensor weights, Tensor gradient) {
        weightsGradients.merge(weights, gradient, Tensor::add);
    }

    /**
     * Resets all accumulated gradients and clears the internal state.
     */
    public void initialize() {
        weightsGradients.clear();
    }

    /**
     * Optional hook called after training on an entire dataset (epoch).
     * @param samples the number of training samples
     */
    public void postFit(double learningRate, int samples) {
        // Overridden by subclasses
    }

    /**
     * Optional hook called after processing a single mini-batch.
     * @param samples      the number of samples in the current batch
     */
    public void postBatch(double learningRate, int samples) {
        // Overridden by subclasses
    }
}
