package org.brain4j.math.convolution.pooling;

import org.brain4j.math.tensor.Tensor;

public abstract class PoolingProvider {

    protected int stride;
    protected int windowHeight;
    protected int windowWidth;

    public PoolingProvider(int stride, int windowHeight, int windowWidth) {
        this.stride = stride;
        this.windowHeight = windowHeight;
        this.windowWidth = windowWidth;
    }

    public abstract Tensor pool(Tensor input);

    public abstract Tensor backward(Tensor gradient, Tensor input);
}
