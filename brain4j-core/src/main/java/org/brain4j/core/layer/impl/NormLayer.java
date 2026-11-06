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
 * Normalizes inputs with learnable scale and shift.
 * Use it to stabilize training in deep or transformer stacks.
 */
public class NormLayer extends Layer {

    /**
     * Holds the numerical stability term.
     */
    public record Config(double epsilon) {}

    protected final Config config;

    /**
     * Create a layer with default epsilon.
     */
    public NormLayer() {
        this(new Config(1e-5));
    }

    /**
     * Create a layer with the given epsilon.
     */
    public NormLayer(double epsilon) {
        this(new Config(epsilon));
    }

    /**
     * Create a layer with the given config.
     */
    public NormLayer(Config config) {
        this.config = config;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void build(List<Shape> inputShapes) {
        Shape inputShape = inputShapes.getFirst();

        Tensor weights = Tensors.ones(inputShape.last());
        Tensor bias = Tensors.zeros(inputShape.last());

        registerParam("weights", weights);
        registerParam("bias", bias);
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
        Tensor first = inputs[0];

        Tensor w = param("weights");
        Tensor b = param("bias");

        // TODO: rework this
        Tensor result = null; // first.layerNormGrad(w, b, config.epsilon);

        return new Tensor[] { result };
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Layer copy() {
        NormLayer copy = new NormLayer(config);
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
