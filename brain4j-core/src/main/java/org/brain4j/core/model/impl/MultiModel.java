package org.brain4j.core.model.impl;

import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.loss.LossFunction;
import org.brain4j.core.merge.MergeStrategy;
import org.brain4j.core.model.Model;
import org.brain4j.core.training.StatesCache;
import org.brain4j.core.training.optimizer.Optimizer;
import org.brain4j.core.training.updater.Updater;
import org.brain4j.core.training.wrappers.EvaluationResult;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.device.DeviceType;
import org.brain4j.math.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.function.Consumer;

/**
 * Represents a composite neural network model that processes multiple independent inputs
 * through separate submodels and merges their outputs using a specified {@link MergeStrategy}
 * before applying a shared sequence of layers for final prediction.
 * <p>
 * This architecture is useful for multi-modal learning tasks where each input may represent
 * different data types requiring distinct preprocessing pipelines.
 * </p>
 *
 * @author xEcho1337
 * @since 3.0
 */
public class MultiModel extends Sequential {

    protected MergeStrategy mergeStrategy;
    protected List<Model> models;

    /**
     * Creates a new {@link MultiModel} instance by combining multiple submodels
     * and a sequence of shared post-merge layers.
     *
     * @param mergeStrategy the strategy used to combine the outputs of submodels
     * @param models the list of submodels, one per input tensor
     * @param layers the layers applied after merging, forming the tail of the network
     * @return a new {@link MultiModel} instance
     */
    public static Sequential of(MergeStrategy mergeStrategy, List<Model> models, Layer... layers) {
        return new MultiModel(mergeStrategy, models, layers);
    }

    protected MultiModel(MergeStrategy mergeStrategy, List<Model> models, Layer... layers) {
        super(layers);
        this.mergeStrategy = mergeStrategy;
        this.models = models;
    }

    @Override
    public Tensor predict(StatesCache cache, boolean training, Tensor... inputs) {
        if (inputs.length != models.size()) {
            throw new IllegalArgumentException("Expected " + models.size() + " inputs, but got " + inputs.length);
        }

        Tensor[] predictions = new Tensor[inputs.length];

        for (int i = 0; i < models.size(); i++) {
            Tensor input = inputs[i];
            Model model = models.get(i);
            predictions[i] = model.predict(cache, training, input);
        }

        Layer first = layers.getFirst();
        Tensor result = mergeStrategy.process(predictions);

        if (!first.validateInput(result)) {
            throw new IllegalArgumentException("Merge strategy output does not match input layer dimension!");
        }

        return super.predict(cache, training, result);
    }

    @Override
    public void backpropagate(StatesCache cache, Tensor outputs, Tensor targets) {
        super.backpropagate(cache, outputs, targets);

        for (Model subModel : models) {
            List<Layer> flattened = subModel.flattened();

            for (int l = flattened.size() - 1; l >= 0; l--) {
                Layer layer = flattened.get(l);

                if (layer.skipPropagate()) continue;

                layer.backward(updater, optimizer, l);
            }
        }
    }

    @Override
    public void updateWeights(Consumer<Layer> callback) {
        for (Layer layer : flattened) {
            callback.accept(layer);
        }

        for (Model subModel : models) {
            List<Layer> flattened = subModel.flattened();

            for (Layer layer : flattened) {
                callback.accept(layer);
            }
        }
    }
}
