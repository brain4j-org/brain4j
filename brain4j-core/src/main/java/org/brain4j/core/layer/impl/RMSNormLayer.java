package org.brain4j.core.layer.impl;

import java.util.List;
import java.util.random.RandomGenerator;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

/**
 * Scales inputs by root-mean-square with learnable weights.
 * Use it as a lighter normalization in transformer blocks.
 */
public class RMSNormLayer extends Layer {

    /**
     * Holds the numerical stability term.
     */
    public record Config(double epsilon) {}

    protected final Config config;

    /**
     * Create a layer with default epsilon.
     */
    public RMSNormLayer() {
        this(new Config(1e-6));
    }

    /**
     * Create a layer with the given epsilon.
     */
    public RMSNormLayer(double epsilon) {
        this(new Config(epsilon));
    }

    /**
     * Create a layer with the given config.
     */
    public RMSNormLayer(Config config) {
        this.config = config;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void build(List<Shape> inputShapes) {
        Shape inputShape = inputShapes.getFirst();
        Tensor weights = Tensors.ones(inputShape.last());
        registerParam("weights", weights);
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
        if (inputShapes.size() != 1) {
            throw Commons.illegalArgument("Layer requires 1 input but %s were given!", inputShapes.size());
        }

        return List.of(inputShapes.getFirst());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];
        Tensor weights = param("weights");

        // TODO: rework this
        Tensor norm = null; // input.rmsNormGrad(weights, config.epsilon);

        return tensors(norm);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Layer copy() {
        RMSNormLayer copy = new RMSNormLayer(config);
        copyParameters(copy);

        return copy;
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
