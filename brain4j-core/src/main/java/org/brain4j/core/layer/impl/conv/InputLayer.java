package org.brain4j.core.layer.impl.conv;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.structure.StatesCache;
import org.brain4j.math.tensor.Tensor;

public class InputLayer extends Layer {

    private final int width;
    private final int height;
    private final int channels;

    public InputLayer(int width, int height, int channels) {
        this.width = width;
        this.height = height;
        this.channels = channels;
    }

    @Override
    public Tensor forward(StatesCache cache, Tensor input, boolean training) {
        return input;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public int getChannels() {
        return channels;
    }
}
