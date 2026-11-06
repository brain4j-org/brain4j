package org.brain4j.core.layer.impl;

import java.util.List;
import java.util.random.RandomGenerator;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

/**
 * Downsamples spatial inputs with max pooling.
 * Use it after convolutions to reduce size while keeping strong activations.
 */
public class MaxPoolLayer extends Layer {

    /**
     * Holds stride and pooling window size.
     */
    public record Config(int stride, int windowHeight, int windowWidth) {}

    protected final Config config;

    /**
     * Create a layer with the given stride and window.
     */
    public MaxPoolLayer(int stride, int windowHeight, int windowWidth) {
        this(new Config(stride, windowHeight, windowWidth));
    }

    /**
     * Create a layer with the given config.
     */
    public MaxPoolLayer(Config config) {
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

        Shape input = inputShapes.getFirst();

        if (input.rank() != 3) {
            throw Commons.illegalArgument("MaxPool requires tensors with rank 3 but %s were given!", input.rank());
        }

        int channels = input.dim(0);
        int height = input.dim(1);
        int width = input.dim(2);

        int outHeight = (height - config.windowHeight) / config.stride + 1;
        int outWidth = (width - config.windowWidth) / config.stride + 1;

        if (outHeight <= 0 || outWidth <= 0) {
            throw Commons.illegalArgument("Negative output dims: outHeight=%s outWidth=%s", outHeight, outWidth);
        }

        return List.of(Shape.of(channels, outHeight, outWidth));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        // TODO: refactor this
        // return tensors(inputs[0].maxPoolGrad(config.stride, config.windowHeight, config.windowWidth));
        return inputs;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Layer copy() {
        return new MaxPoolLayer(config);
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
