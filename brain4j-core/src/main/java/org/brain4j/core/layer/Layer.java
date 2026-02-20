package org.brain4j.core.layer;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.Linear;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.weightsinit.WeightInit;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.random.RandomGenerator;

public abstract class Layer {
    
    protected Map<String, Tensor> parameters;
    protected Activation activation;
    protected WeightInit weightInit;
    
    public Layer() {
        this(new Linear());
    }
    
    public Layer(Activation activation) {
        this.parameters = new HashMap<>();
        this.activation = activation;
        this.weightInit = activation.defaultWeightInit();
    }
    
    public abstract void build(List<Shape> inputShapes, RandomGenerator rng);
    
    public abstract List<Shape> inferOutputShapes(List<Shape> inputShapes);
    
    public abstract Tensor[] forward(StatesCache cache, Tensor... inputs);
    
    public void generateWeights(RandomGenerator rng, int input, int output) {
        for (Tensor parameter : parameters.values()) {
            parameter.map(x -> weightInit.generate(rng, input, output));
        }
    }
    
    public Node apply(Node... inputs) {
        return new Node(this, List.of(inputs));
    }
    
    public Tensor getParam(String name) {
        return parameters.get(name);
    }
    
    public Map<String, Tensor> parameters() {
        return parameters;
    }
}