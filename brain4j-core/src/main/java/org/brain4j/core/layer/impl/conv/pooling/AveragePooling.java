package org.brain4j.core.layer.impl.conv.pooling;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.structure.StatesCache;
import org.brain4j.math.tensor.Tensor;

public class AveragePooling extends Layer {

    private final int width;
    private final int height;
    private final int padding;

    public AveragePooling(int width, int height) {
        this(width, height, 1);
    }

    public AveragePooling(int width, int height, int padding) {
        this.width = width;
        this.height = height;
        this.padding = padding;
    }

    @Override
    public boolean canConnect() {
        return false;
    }

    @Override
    public Tensor forward(int index, StatesCache cache, Tensor input, boolean training) {
        return null;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public int getPadding() {
        return padding;
    }
}
