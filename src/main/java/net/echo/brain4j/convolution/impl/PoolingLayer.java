package net.echo.brain4j.convolution.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.convolution.pooling.PoolingType;
import net.echo.brain4j.layer.Layer;

public class PoolingLayer extends Layer {

    protected final PoolingType poolingType;
    protected final int kernelWidth;
    protected final int kernelHeight;
    protected int stride;
    protected int padding;

    public PoolingLayer(PoolingType poolingType, int kernelWidth, int kernelHeight) {
        this(poolingType, kernelWidth, kernelHeight, 1, 0);
    }

    public PoolingLayer(PoolingType poolingType, int kernelWidth, int kernelHeight, int stride) {
        this(poolingType, kernelWidth, kernelHeight, stride, 0);
    }

    public PoolingLayer(PoolingType poolingType, int kernelWidth, int kernelHeight, int stride, int padding) {
        super(kernelWidth * kernelHeight, Activations.LINEAR);
        this.poolingType = poolingType;
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.stride = stride;
        this.padding = padding;
    }
}
