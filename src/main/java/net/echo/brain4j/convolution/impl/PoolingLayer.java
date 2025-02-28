package net.echo.brain4j.convolution.impl;

import com.google.common.base.Preconditions;
import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.convolution.Kernel;
import net.echo.brain4j.convolution.pooling.PoolingType;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.cache.StatesCache;

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

    @Override
    public boolean isConvolutional() {
        return true;
    }

    @Override
    public Kernel forward(StatesCache cache, Layer layer, Kernel input) {
        Preconditions.checkNotNull(input, "Last convolutional input is null");

        double initialWidth = input.getWidth() - kernelWidth + 2 * padding;
        double initialHeight = input.getHeight() - kernelHeight + 2 * padding;

        int outputWidth = (int) Math.ceil(initialWidth / stride) + 1;
        int outputHeight = (int) Math.ceil(initialHeight / stride) + 1;

        Kernel output = new Kernel(outputWidth, outputHeight);

        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < outputWidth; j++) {
                double value = poolingType.getFunction().apply(this, input, i, j);
                output.setValue(j, i, value);
            }
        }

        return output;
    }

    public PoolingType getPoolingType() {
        return poolingType;
    }

    public int getKernelWidth() {
        return kernelWidth;
    }

    public int getKernelHeight() {
        return kernelHeight;
    }

    public int getStride() {
        return stride;
    }

    public int getPadding() {
        return padding;
    }
}
