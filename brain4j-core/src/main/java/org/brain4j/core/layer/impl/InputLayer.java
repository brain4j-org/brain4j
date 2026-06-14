package org.brain4j.core.layer.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;
import org.brain4j.math.commons.Commons;

public class InputLayer extends Layer {

    public record Config(Shape shape) {}

    protected Config config;

    public InputLayer(Shape shape) {
        this(new Config(shape));
    }

    public InputLayer(Config config) {
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
        return List.of(config.shape);
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        for (Tensor input : inputs) {
            if (validInput(input)) continue;

            throw Commons.illegalArgument("Input must have shape %s! Got: %s",
                java.util.Arrays.toString(config.shape.dims()), java.util.Arrays.toString(input.shape()));
        }

        return inputs;
    }

    @Override
    public Layer copy() {
        return new InputLayer(config.shape.copy());
    }

    public Config config() {
        return config;
    }

    private boolean validInput(Tensor input) {
        if (input == null) return false;

        int[] inputShape = input.shape();
        int[] targetShape = config.shape.dims();

        if (inputShape.length - 1 > targetShape.length) return false;

        int offset = inputShape.length - targetShape.length;

        for (int i = 0; i < targetShape.length; i++) {
            if (inputShape[i + offset] != targetShape[i]) return false;
        }

        return true;
    }
}
