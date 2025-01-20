package net.echo.brain4j.convolution.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.Layer;

public class ConvolutionalLayer extends Layer {

    protected final int kernelWidth;
    protected final int kernelHeight;
    protected final int filters;
    protected int padding;
    protected int stride;

    public ConvolutionalLayer(int filters, int kernelWidth, int kernelHeight, Activations activation) {
        this(filters, kernelWidth, kernelHeight, 1, 0, activation);
    }

    public ConvolutionalLayer(int filters, int kernelWidth, int kernelHeight, int stride, Activations activation) {
        this(filters, kernelWidth, kernelHeight, stride, 0, activation);
    }

    public ConvolutionalLayer(int filters, int kernelWidth, int kernelHeight, int stride, int padding, Activations activation) {
        super(0, activation);
        this.filters = filters;
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;
        this.stride = stride;
        this.padding = padding;
    }

    public int getKernelWidth() {
        return kernelWidth;
    }

    public int getKernelHeight() {
        return kernelHeight;
    }

    public int getFilters() {
        return filters;
    }

    public int getPadding() {
        return padding;
    }

    public int getStride() {
        return stride;
    }
}
