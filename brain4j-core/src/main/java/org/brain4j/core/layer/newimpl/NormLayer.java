package org.brain4j.core.layer.newimpl;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

public class NormLayer extends Layer {
    
    private final double epsilon;
    
    public NormLayer() {
        this(1e-5);
    }
    
    public NormLayer(double epsilon) {
        this.epsilon = epsilon;
    }
    
    @Override
    public void build(List<Shape> inputShapes) {
        Shape inputShape = inputShapes.getFirst();
        
        Tensor weights = Tensors.ones(inputShape.last());
        Tensor bias = Tensors.zeros(inputShape.last());
    
        parameters.put("weights", weights);
        parameters.put("bias", bias);
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
        Tensor first = inputs[0];
        Tensor cloned = first.copy();
        
        cloned.setAutogradContext(first.getAutogradContext());
        
        Tensor W = getParam("weights");
        Tensor B = getParam("bias");
        
        Tensor result = cloned.layerNorm(epsilon)
            .mulGrad(W)
            .addGrad(B);
        
        return new Tensor[] { result };
    }
    
    @Override
    public Layer copy() {
        NormLayer copy = new NormLayer(epsilon);
        copyParameters(copy);
        return copy;
    }
    
    public double epsilon() {
        return epsilon;
    }
}
