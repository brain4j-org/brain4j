package org.brain4j.core.layer.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

public class MaxPoolLayer extends Layer {

    public record Config(int stride, int windowHeight, int windowWidth) {}

    protected final Config config;

    public MaxPoolLayer(int stride, int windowHeight, int windowWidth) {
        this(new Config(stride, windowHeight, windowWidth));
    }

    public MaxPoolLayer(Config config) {
        this.config = config;
    }

    @Override
    public void build(List<Shape> inputShapes) {
    }

    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
    }

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

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        return tensors(inputs[0].maxPoolGrad(config.stride, config.windowHeight, config.windowWidth));
    }

    @Override
    public Layer copy() {
        return new MaxPoolLayer(config);
    }

    public Config config() {
        return config;
    }
}
