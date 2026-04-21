package org.brain4j.core.layer.impl;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;
import org.brain4j.math.commons.Commons;

public class InputLayer extends Layer {
    
    private final Shape shape;
    
    public InputLayer(Shape shape) {
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
        return List.of(shape);
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        for (Tensor input : inputs) {
            if (validInput(input)) continue;

            throw Commons.illegalArgument("Input must have shape %s! Got: %s",
                java.util.Arrays.toString(shape.dims()), java.util.Arrays.toString(input.shape()));
        }

        return inputs;
    }
    
    @Override
    public Layer copy() {
        return new InputLayer(shape.copy());
    }
    
    public Shape shape() {
        return shape;
    }
    
    private boolean validInput(Tensor input) {
        if (input == null) return false;
        
        int[] inputShape = input.shape();
        int[] targetShape = shape.dims();
        
        if (inputShape.length - 1 > targetShape.length) return false;
        
        int offset = inputShape.length - targetShape.length;
        
        for (int i = 0; i < targetShape.length; i++) {
            if (inputShape[i + offset] != targetShape[i]) return false;
        }

        return true;
    }
}
