package org.brain4j.core.layer.impl;

import java.util.List;
import java.util.random.RandomGenerator;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.Linear;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

/**
 * Applies an affine projection with bias and activation.
 * Use it as the standard fully connected block in MLPs and heads.
 */
public class DenseLayer extends Layer {

    /**
     * Holds the output size and activation.
     */
    public record Config(int outDimension, Activation activation) {}

    protected final Config config;

    /**
     * Create a layer with linear activation.
     */
    public DenseLayer(int outDimension) {
        this(new Config(outDimension, new Linear()));
    }

    /**
     * Create a layer with the given activation.
     */
    public DenseLayer(int outDimension, Activation activation) {
        this(new Config(outDimension, activation));
    }

    /**
     * Create a layer with the given config.
     */
    public DenseLayer(Config config) {
        super(config.activation);
        this.config = config;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void build(List<Shape> inputShapes) {
        Shape inputShape = inputShapes.getFirst();

        Tensor weights = Tensors.zeros(inputShape.last(), config.outDimension);
        Tensor bias = Tensors.zeros(config.outDimension);

        registerParam("weights", weights);
        registerParam("bias", bias);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
        generateWeights("weights", rng, inputShapes.getFirst().last(), config.outDimension);
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
        Shape inputBatch = inputShape.slice(0, -2);

        int[] outShape = new int[inputShape.rank()];

        inputBatch.copy(outShape);
        outShape[outShape.length - 1] = config.outDimension;

        return List.of(Shape.of(outShape));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];

        if (input.rank() < 2) {
            throw Commons.illegalArgument("Dense requires tensors to be rank 2 or higher");
        }

        Tensor w = param("weights");
        Tensor b = param("bias");

        Tensor proj = input.matmul(w).add(b);

        return tensors(proj.activate(activation));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DenseLayer copy() {
        DenseLayer copy = new DenseLayer(config);
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
