package org.brain4j.core.layer.newimpl.utility;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

public class SelectLayer extends Layer {
    
    private final int index;
    
    public SelectLayer(int index) {
        this.index = index;
    }

    @Override
    public void build(List<Shape> inputShapes) {
    }

    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
    }

    @Override
    public List<Shape> inferOutputShapes(List<Shape> inputShapes) {
        if (inputShapes.isEmpty()) {
            throw Commons.illegalArgument("Layer requires at least 1 input but 0 were given!");
        }
        
        if (index < 0 || index >= inputShapes.size()) {
            throw Commons.illegalArgument("Selection index %s is out of range (size=%s)", index, inputShapes.size());
        }
        
        return List.of(inputShapes.get(index));
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        return tensors(inputs[index]);
    }

    @Override
    public Layer copy() {
        return new SelectLayer(index);
    }
    
    public int index() {
        return index;
    }
}
