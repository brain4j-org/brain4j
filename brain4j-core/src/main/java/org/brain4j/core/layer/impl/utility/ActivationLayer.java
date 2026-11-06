package org.brain4j.core.layer.impl.utility;

import java.util.List;
import java.util.random.RandomGenerator;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

/**
 * Applies an element-wise activation function.
 * Use it to add non-linearity without changing shapes.
 */
public class ActivationLayer extends Layer {

    /**
     * Holds the activation function.
     */
    public record Config(Activation activation) {}

    protected final Config config;

    /**
     * Create a layer with the given activation enum.
     */
    public ActivationLayer(Activations activation) {
        this(activation.function());
    }

    /**
     * Create a layer with the given activation.
     */
    public ActivationLayer(Activation activation) {
        this(new Config(activation));
    }

    /**
     * Create a layer with the given config.
     */
    public ActivationLayer(Config config) {
        super(config.activation);
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
        if (inputShapes.isEmpty()) {
            throw Commons.illegalArgument("Layer requires at least 1 input but 0 were given!");
        }

        return inputShapes;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor[] result = new Tensor[inputs.length];

        for (int i = 0; i < inputs.length; i++) {
            result[i] = inputs[i].activate(activation);
        }

        return result;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Layer copy() {
        return new ActivationLayer(config);
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
