package org.brain4j.core.layer.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

public class DropoutLayer extends Layer {

    public record Config(double dropoutRate) {}

    private final RandomGenerator random;
    protected Config config;

    public DropoutLayer(double dropoutRate) {
        this(new Config(dropoutRate));
    }

    public DropoutLayer(Config config) {
        if (config.dropoutRate < 0 || config.dropoutRate >= 1) {
            throw Commons.illegalArgument("Dropout must be greater or equal to 0 and less than 1!");
        }

        this.random = new SplittableRandom();
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
        return inputShapes;
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        if (!cache.isTraining()) return inputs;

        Tensor[] result = new Tensor[inputs.length];

        for (int i = 0; i < inputs.length; i++) {
            Tensor input = inputs[i];
            float[] mask = new float[input.elements()];

            for (int j = 0; j < mask.length; j++) {
                mask[j] = random.nextFloat() > config.dropoutRate ? 1 : 0;
            }

            Tensor tensorMask = Tensors.vector(mask);
            Tensor reshaped = input.reshapeGrad(input.elements());

            result[i] = reshaped.mulGrad(tensorMask)
                .div(1 - config.dropoutRate)
                .reshapeGrad(input.shape());
        }

        return result;
    }

    @Override
    public DropoutLayer copy() {
        return new DropoutLayer(config);
    }

    public RandomGenerator random() {
        return random;
    }

    public Config config() {
        return config;
    }
}
