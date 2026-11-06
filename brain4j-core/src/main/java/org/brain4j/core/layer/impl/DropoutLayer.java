package org.brain4j.core.layer.impl;

import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

/**
 * Randomly zeroes inputs during training with inverted scaling.
 * Use it between dense layers to reduce overfitting.
 */
public class DropoutLayer extends Layer {

    /**
     * Holds the dropout probability.
     */
    public record Config(double dropoutRate) {}

    private final RandomGenerator random;
    protected Config config;

    /**
     * Create a layer with the given dropout rate.
     *
     * @param dropoutRate the drop probability in [0, 1)
     */
    public DropoutLayer(double dropoutRate) {
        this(new Config(dropoutRate));
    }

    /**
     * Create a layer with the given config.
     */
    public DropoutLayer(Config config) {
        if (config.dropoutRate < 0 || config.dropoutRate >= 1) {
            throw Commons.illegalArgument("Dropout must be greater or equal to 0 and less than 1!");
        }

        this.random = new SplittableRandom();
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
        if (!cache.isTraining()) {
            return inputs;
        }

        Tensor[] result = new Tensor[inputs.length];

        for (int i = 0; i < inputs.length; i++) {
            Tensor input = inputs[i];
            float[] mask = new float[input.elements()];

            for (int j = 0; j < mask.length; j++) {
                mask[j] = random.nextFloat() > config.dropoutRate ? 1 : 0;
            }

            Tensor tensorMask = Tensors.vector(mask);
            Tensor reshaped = input.reshape(input.elements());

            result[i] = reshaped.mul(tensorMask)
                .div(1 - config.dropoutRate)
                .reshape(input.shape());
        }

        return result;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DropoutLayer copy() {
        return new DropoutLayer(config);
    }

    /**
     * Return the random source.
     *
     * @return the random generator
     */
    public RandomGenerator random() {
        return random;
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
