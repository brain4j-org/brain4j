package org.brain4j.core.layer.newimpl;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.Linear;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.List;

public class DenseLayer extends Layer {
    
    private int outDimension;
    
    public DenseLayer(int outDimension) {
        this(outDimension, new Linear());
    }
    
    public DenseLayer(int outDimension, Activation activation) {
        this.outDimension = outDimension;
        this.activation = activation;
    }
    
    @Override
    public void build(List<Shape> inputShapes) {
        inferOutputShapes(inputShapes);
        
        Shape inputShape = inputShapes.getFirst();
        
        Tensor weights = Tensors.matrix(inputShape.last(), outDimension);
        Tensor bias = Tensors.vector(outDimension);
        
        parameters.put("weights", weights);
        parameters.put("bias", bias);
    }
    
    @Override
    public List<Shape> inferOutputShapes(List<Shape> inputShapes) {
        if (inputShapes.size() != 1) {
            throw Commons.illegalArgument("Dense requires 1 input but %s were given!", inputShapes.size());
        }
        
        Shape inputShape = inputShapes.getFirst();
        
        if (inputShape.rank() < 2) {
            throw Commons.illegalArgument("Dense requires tensors to be rank 2 or higher");
        }
        
        Shape inputBatch = inputShape.slice(0, -2);
        
        int[] outShape = new int[inputShape.rank()];
        
        inputBatch.copy(outShape);
        outShape[outShape.length - 1] = outDimension;
        
        return List.of(Shape.of(outShape));
    }
    
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];
        
        cache.setStates(this, "input", input);
        
        Tensor W = getParam("weights");
        Tensor B = getParam("bias");
        
        Tensor proj = input.matmulGrad(W).addGrad(B);
        cache.setStates(this, "pre_activation", proj);
        
        return new Tensor[] { proj.activateGrad(activation) };
    }
}
