package net.echo.brain4j.convolution.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.convolution.Kernel;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.cache.StatesCache;

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

    public Kernel getImage(StatesCache cache) {
        Kernel result = new Kernel(width, height);

        for (int x = 0; x < width; x++) {
            for (int h = 0; h < height; h++) {
                Neuron neuron = getNeurons().get(h * width + x);

                result.getValues()[h].set(x, neuron.getValue(cache));
            }
        }

        return result;
    }
}
