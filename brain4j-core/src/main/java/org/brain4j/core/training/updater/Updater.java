package org.brain4j.core.training.updater;

import java.util.HashMap;
import java.util.Map;

import org.brain4j.core.training.updater.impl.NormalUpdater;
import org.brain4j.core.training.updater.impl.StochasticUpdater;
import org.brain4j.math.Tensors;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.impl.GpuTensor;

/**
 * Accumulates updates and applies them to weights.
 * Use it to define when batched updates modify the model.
 *
 * @see StochasticUpdater
 * @see NormalUpdater
 */
public abstract class Updater {

    protected final Map<Tensor, Tensor> weightsUpdates = new HashMap<>();

    /**
     * Persistent GPU accumulation buffers reused across batches.
     */
    protected final Map<Tensor, Tensor> persistentStorage = new HashMap<>();

    /**
     * Applies accumulated updates to weights in place.
     */
    public void updateWeights() {
        for (Map.Entry<Tensor, Tensor> entry : weightsUpdates.entrySet()) {
            Tensor weights = entry.getKey();
            Tensor update = entry.getValue();

            if (update == null || weights == null) {
                continue;
            }

            while (update.rank() > weights.rank()) {
                update = update.sum(0, false);
            }
z
            weights.sub(update);
        }
    }

    /**
     * Accumulates an update for the given weights.
     */
    public void change(Tensor weights, Tensor update) {
        if (!(update instanceof GpuTensor gpuUpdate)) {
            weightsUpdates.merge(weights, update, Tensor::add);
            return;
        }

        Tensor stored = weightsUpdates.get(weights);

        if (stored == null) {
            stored = persistentStorage.get(weights);

            if (stored == null) {
                stored = GpuTensor.persistent(Tensors.zeros(update.shape()), gpuUpdate.device());
                persistentStorage.put(weights, stored);
            } else {
                // Reuse the buffer instead of leaking GPU memory.
                stored.mul(0);
            }
        }

        stored.add(update);
        weightsUpdates.put(weights, stored);
    }

    /**
     * Clears all accumulated updates.
     */
    public void initialize() {
        weightsUpdates.clear();
    }

    /**
     * Runs a hook after training over the dataset.
     */
    public void postFit() {
    }

    /**
     * Runs a hook after each mini-batch.
     */
    public void postBatch() {
    }
}
