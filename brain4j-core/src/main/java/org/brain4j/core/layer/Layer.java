package org.brain4j.core.layer;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.Linear;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public abstract class Layer {
    
    protected Map<String, Tensor> parameters;
    protected Activation activation;
    
    public Layer() {
        this(new Linear());
    }
    
    public Layer(Activation activation) {
        this.parameters = new HashMap<>();
        this.activation = activation;
    }
    
    public abstract void build(List<Shape> inputShapes);
    
    public abstract List<Shape> inferOutputShapes(List<Shape> inputShapes);
    
    public abstract Tensor[] forward(StatesCache cache, Tensor... inputs);
    
    public Tensor getParam(String name) {
        return parameters.get(name);
    }
    
    public Map<String, Tensor> parameters() {
        return parameters;
    }
}