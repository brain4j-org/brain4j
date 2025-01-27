package net.echo.brain4j.convolution.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;

public class FlattenLayer extends Layer {

    public FlattenLayer() {
        super(0, Activations.LINEAR);
    }
}
