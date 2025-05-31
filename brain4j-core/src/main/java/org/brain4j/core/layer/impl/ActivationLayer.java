package org.brain4j.core.layer.impl;

import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.tensor.Tensor;

public class ActivationLayer extends Layer {

    private final Activations activation;
    private int dimension;

    public ActivationLayer(Activations activation) {
        this.activation = activation;
    }

    @Override
    public void connect(Layer previous) {
        this.dimension = previous.size();
    }

    @Override
    public Tensor forward(ForwardContext context) {
        Tensor input = context.input();
        StatesCache cache = context.cache();

        Tensor output = input.activateGrad(activation.getFunction());

        cache.setInput(context.index(), input);
        cache.setOutput(context.index(), output);

        return output;
    }

    @Override
    public int size() {
        return dimension;
    }
}
