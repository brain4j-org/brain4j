package net.echo.brain4j.convolution.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;

public class FlattenLayer extends Layer {

    public FlattenLayer(int input) {
        super(input, Activations.LINEAR);
    }
}
