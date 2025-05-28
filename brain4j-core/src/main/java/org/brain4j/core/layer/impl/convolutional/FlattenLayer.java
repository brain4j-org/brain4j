package org.brain4j.core.layer.impl.convolutional;

import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
import org.brain4j.math.activation.impl.LinearActivation;
import org.brain4j.math.tensor.Tensor;

public class FlattenLayer extends Layer {

    private int dimension;

    public FlattenLayer() {
        super(new LinearActivation(), null);
    }

    @Override
    public void connect(Layer previous, Layer next) {
        this.dimension = previous.size();
    }

    @Override
    public Tensor forward(ForwardContext context) {
        return null;
    }

    @Override
    public int size() {
        return 0;
    }
}
