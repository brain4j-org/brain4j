package org.brain4j.core.layer.newimpl;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

public class ReshapeLayer extends Layer {
    
    private final Shape shape;
    
    public ReshapeLayer(Shape shape) {
        this.shape = shape;
    }
    
    @Override
    public void build(List<Shape> inputShapes) {
    }
    
    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
    }
    
    @Override
    public List<Shape> inferOutputShapes(List<Shape> inputShapes) {
        if (inputShapes.size() != 1) {
            throw Commons.illegalArgument("Layer requires 1 input but %s were given!", inputShapes.size());
        }
        
        Shape inputShape = inputShapes.getFirst();
        
        int inputSize = inputShape.size();
        int newSize = shape.size();
        
        if (inputSize != newSize) {
            throw Commons.illegalArgument("Input size (%s) does not match reshape size (%s)", inputSize, newSize);
        }
        
        return List.of(shape);
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];
        
        int[] inputShape = input.shape();
        int[] newShape = new int[shape.rank() + 1];
        
        newShape[0] = inputShape[0];
        System.arraycopy(shape.dims(), 0, newShape, 1, shape.rank());
        
        return tensors(inputs[0].reshapeGrad(newShape));
    }
    
    @Override
    public Layer copy() {
        return new ReshapeLayer(shape);
    }
}
