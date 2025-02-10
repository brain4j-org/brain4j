package net.echo.brain4j.convolution.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;

public class InputLayer extends Layer {

    private final int width;
    private final int height;

    public InputLayer(int width, int height) {
        super(width * height, Activations.LINEAR);
        this.width = width;
        this.height = height;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    @Override
    public boolean isConvolutional() {
        return true;
    }

    @Override
    public int size() {
        return width * height;
    }
}
