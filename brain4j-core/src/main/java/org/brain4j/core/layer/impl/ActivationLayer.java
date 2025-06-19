package org.brain4j.core.layer.impl;

import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.tensor.Tensor;

public class ActivationLayer extends Layer {

    private int dimension;

    public ActivationLayer(Activations activation) {
        this.activation = activation.function();
    }

    @Override
    public Layer connect(Layer previous) {
        this.dimension = previous.size();
        return this;
    }

    @Override
    public Tensor forward(ForwardContext context) {
        Tensor input = context.input();
        StatesCache cache = context.cache();

        cache.setPreActivation(this, input);
        return input.activateGrad(activation);
    }

    @Override
    public int size() {
        return dimension;
    }
}
