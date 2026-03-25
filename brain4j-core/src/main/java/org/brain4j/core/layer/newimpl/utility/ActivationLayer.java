package org.brain4j.core.layer.newimpl.utility;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

public class ActivationLayer extends Layer {

    public ActivationLayer(Activations activation) {
        this(activation.function());
    }
    
    public ActivationLayer(Activation activation) {
        super(activation);
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
        
        return inputShapes;
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor[] result = new Tensor[inputs.length];
        
        for (int i = 0; i < inputs.length; i++) {
            result[i] = inputs[i].activateGrad(activation);
        }
        
        return result;
    }

    @Override
    public Layer copy() {
        return new ActivationLayer(activation);
    }
}
