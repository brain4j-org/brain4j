package org.brain4j.core.layer.impl.transformer;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.weightsinit.impl.UniformXavierInit;

import java.util.List;
import java.util.random.RandomGenerator;

public class EmbeddingLayer extends Layer {

    public record Config(int vocabSize, int embeddingDim) {}

    protected final Config config;

    public EmbeddingLayer(int vocabSize, int embeddingDim) {
        this(new Config(vocabSize, embeddingDim));
    }

    public EmbeddingLayer(Config config) {
        this.config = config;
        this.weightInit = new UniformXavierInit();
    }

    @Override
    public void build(List<Shape> inputShapes) {
        registerParam("weights", Tensors.zeros(config.vocabSize, config.embeddingDim));
    }

    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
        generateWeights("weights", rng, config.vocabSize, config.embeddingDim);
    }

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

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];
        int[] shape = input.shape();

        if (shape.length != 2) {
            throw Commons.illegalState("Input must have shape [batch, seq_length]! Got: %s",
                java.util.Arrays.toString(shape));
        }

        Tensor output = input.gatherGrad(getParam("weights"));

        return new Tensor[] { output };
    }

    @Override
    public Layer copy() {
        EmbeddingLayer copy = new EmbeddingLayer(config);
        copyParameters(copy);
        return copy;
    }

    public Config config() {
        return config;
    }
}
