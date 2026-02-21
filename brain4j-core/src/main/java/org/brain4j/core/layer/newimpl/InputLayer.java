package org.brain4j.core.layer.newimpl;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

public class InputLayer extends Layer {
    
    private final Shape shape;
    
    public InputLayer(Shape shape) {
        this.shape = shape;
    }
    
    @Override
    public void build(List<Shape> inputShapes, RandomGenerator rng) {
    }
    
    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
    }
    
    @Override
    public List<Shape> inferOutputShapes(List<Shape> inputShapes) {
        return List.of(shape);
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        return inputs;
    }
    
    public Shape shape() {
        return shape;
    }
    
    @Override
    public Layer copy() {
        return new InputLayer(shape.copy());
    }
}
