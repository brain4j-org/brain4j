package org.brain4j.core.layer.impl.transformer;

import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.weightsinit.impl.UniformXavierInit;

/**
 * Maps token ids to dense embeddings.
 * Use it as the first block of transformer models.
 */
public class EmbeddingLayer extends Layer {

    /**
     * Holds vocabulary size and embedding width.
     */
    public record Config(int vocabSize, int embeddingDim) {}

    protected final Config config;

    /**
     * Create a layer with the given vocabulary and width.
     */
    public EmbeddingLayer(int vocabSize, int embeddingDim) {
        this(new Config(vocabSize, embeddingDim));
    }

    /**
     * Create a layer with the given config.
     */
    public EmbeddingLayer(Config config) {
        this.config = config;
        this.weightInit = new UniformXavierInit();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void build(List<Shape> inputShapes) {
        registerParam("weights", Tensors.zeros(config.vocabSize, config.embeddingDim));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
        generateWeights("weights", rng, config.vocabSize, config.embeddingDim);
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

        if (inputShape.rank() != 1) {
            throw Commons.illegalArgument("Embedding requires tensors with rank 1 but %s were given!", inputShape.rank());
        }

        return List.of(Shape.of(inputShape.dim(0), config.embeddingDim));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];
        int[] shape = input.shape();

        if (shape.length != 2) {
            throw Commons.illegalState("Input must have shape [batch, seq_length]! Got: %s",
                Arrays.toString(shape));
        }

        Tensor output = input.gather(param("weights"));

        return new Tensor[] { output };
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Layer copy() {
        EmbeddingLayer copy = new EmbeddingLayer(config);
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
