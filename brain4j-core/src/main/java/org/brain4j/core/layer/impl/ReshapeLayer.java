package org.brain4j.core.layer.impl;

import java.util.List;
import java.util.random.RandomGenerator;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

/**
 * Reshapes inputs to a fixed shape while preserving batch size.
 * Use it to adapt tensor layouts between layers.
 */
public class ReshapeLayer extends Layer {

    /**
     * Holds the target shape without batch dim.
     */
    public record Config(Shape shape) {}

    protected final Config config;

    /**
     * Create a layer with the given target shape.
     */
    public ReshapeLayer(Shape shape) {
        this(new Config(shape));
    }

    /**
     * Create a layer with the given config.
     */
    public ReshapeLayer(Config config) {
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
        if (inputShapes.size() != 1) {
            throw Commons.illegalArgument("Layer requires 1 input but %s were given!", inputShapes.size());
        }

        Shape inputShape = inputShapes.getFirst();

        int inputSize = inputShape.size();
        int newSize = config.shape.size();

        if (inputSize != newSize) {
            throw Commons.illegalArgument("Input size (%s) does not match reshape size (%s)", inputSize, newSize);
        }

        return List.of(config.shape);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];

        int[] inputShape = input.shape();
        int[] newShape = new int[config.shape.rank() + 1];

        newShape[0] = inputShape[0];
        System.arraycopy(config.shape.dims(), 0, newShape, 1, config.shape.rank());

        return tensors(inputs[0].reshape(newShape));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Layer copy() {
        return new ReshapeLayer(config.shape);
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
