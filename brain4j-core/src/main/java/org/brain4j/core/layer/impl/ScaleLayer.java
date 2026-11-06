package org.brain4j.core.layer.impl;

import java.util.List;
import java.util.Set;
import java.util.random.RandomGenerator;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.scaler.FeatureScaler;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.operations.AutogradContext;

/**
 * Applies feature scaling without changing shapes.
 * Use it to normalize inputs before training or inference.
 */
public class ScaleLayer extends Layer {

    /**
     * Holds the scaler and the enabled input indexes.
     */
    public record Config(FeatureScaler scaler, Set<Integer> enabledInputs) {}

    protected final Config config;

    /**
     * Create a layer scaling all inputs.
     */
    public ScaleLayer(FeatureScaler scaler) {
        this(new Config(scaler, null));
    }

    /**
     * Create a layer scaling only the selected inputs.
     *
     * @param enabledInputs the input indexes to scale, {@code null} scales all
     */
    public ScaleLayer(FeatureScaler scaler, Set<Integer> enabledInputs) {
        this(new Config(scaler, enabledInputs));
    }

    /**
     * Create a layer with the given config.
     */
    public ScaleLayer(Config config) {
        this.config = config;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void build(List<Shape> inputShapes) {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<Shape> inferOutputShapes(List<Shape> inputShapes) {
        return inputShapes;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        boolean scaleAll = config.enabledInputs == null;

        if (!scaleAll) {
            for (int i : config.enabledInputs) {
                if (i < 0 || i >= inputs.length) {
                    throw Commons.illegalState("Enabled input index out of range: %s", i);
                }
            }
        }

        Tensor[] outputs = new Tensor[inputs.length];

        for (int i = 0; i < outputs.length; i++) {
            Tensor input = inputs[i];

            if (!scaleAll && !config.enabledInputs.contains(i)) {
                outputs[i] = input;
                continue;
            }

            Tensor result = config.scaler.transform(input);

            AutogradContext context = input.autogradContext();
            result.setAutogradContext(context);

            outputs[i] = result;
        }

        return outputs;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Layer copy() {
        return new ScaleLayer(config);
    }

    /**
     * Return the layer config.
     *
     * @return the config
     */
    public Config config() {
        return config;
    }
}
