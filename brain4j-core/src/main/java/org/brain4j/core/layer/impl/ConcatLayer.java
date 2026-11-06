package org.brain4j.core.layer.impl;

import java.util.List;
import java.util.random.RandomGenerator;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

/**
 * Concatenates inputs along one dimension.
 * Use it to merge parallel branches before downstream layers.
 */
public class ConcatLayer extends Layer {

    /**
     * Holds the concat dimension.
     */
    public record Config(int dimension) {}

    protected final Config config;

    /**
     * Create a layer concatenating on the last dimension.
     */
    public ConcatLayer() {
        this(new Config(-1));
    }

    /**
     * Create a layer concatenating on the given dimension.
     */
    public ConcatLayer(int dimension) {
        this(new Config(dimension));
    }

    /**
     * Create a layer with the given config.
     */
    public ConcatLayer(Config config) {
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
        return List.of(Shape.concat(inputShapes.toArray(new Shape[0])));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        return new Tensor[] { Tensors.concat(List.of(inputs), config.dimension) };
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Layer copy() {
        return new ConcatLayer(config);
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
