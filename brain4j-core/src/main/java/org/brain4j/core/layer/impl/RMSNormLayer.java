package org.brain4j.core.layer.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

public class RMSNormLayer extends Layer {

    public record Config(double epsilon) {}

    protected final Config config;

    public RMSNormLayer() {
        this(new Config(1e-6));
    }

    public RMSNormLayer(double epsilon) {
        this(new Config(epsilon));
    }

    public RMSNormLayer(Config config) {
        this.config = config;
    }

    @Override
    public void build(List<Shape> inputShapes) {
        Shape inputShape = inputShapes.getFirst();
        Tensor weights = Tensors.ones(inputShape.last());
        registerParam("weights", weights);
    }

    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
    }

    @Override
    public List<Shape> inferOutputShapes(List<Shape> inputShapes) {
        if (inputShapes.size() != 1) {
            throw Commons.illegalArgument("Layer requires 1 input but %s were given!", inputShapes.size());
        }

        return List.of(inputShapes.getFirst());
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];
        Tensor weights = getParam("weights");

        Tensor norm = input.rmsNormGrad(weights, config.epsilon);
        return tensors(norm);
    }

    @Override
    public Layer copy() {
        RMSNormLayer copy = new RMSNormLayer(config);
        copyParameters(copy);
        return copy;
    }

    public Config config() {
        return config;
    }
}
