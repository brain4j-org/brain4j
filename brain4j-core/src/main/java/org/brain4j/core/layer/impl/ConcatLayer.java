package org.brain4j.core.layer.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

public class ConcatLayer extends Layer {
    
    private final int dimension;
    
    public ConcatLayer() {
        this(-1);
    }
    
    public ConcatLayer(int dimension) {
        this.dimension = dimension;
    }
    
    @Override
    public void build(List<Shape> inputShapes) {
    }
    
    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
    }
    
    @Override
    public List<Shape> inferOutputShapes(List<Shape> inputShapes) {
        return List.of(Shape.concat(inputShapes.toArray(new Shape[0])));
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        return new Tensor[] { Tensors.concatGrad(List.of(inputs), dimension) };
    }
    
    @Override
    public Layer copy() {
        return new ConcatLayer(dimension);
    }
    
    public int dimension() {
        return dimension;
    }
}
