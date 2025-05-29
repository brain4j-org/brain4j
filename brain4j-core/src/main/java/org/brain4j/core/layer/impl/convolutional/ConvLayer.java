package org.brain4j.core.layer.impl.convolutional;

import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.tensor.Tensor;

public class ConvLayer extends Layer {

    private int filters;
    private int kernelWidth;
    private int kernelHeight;
    private int channels;
    private int stride;
    private int padding;

    public ConvLayer(Activations activation, int filters, int kernelWidth, int kernelHeight) {
        this.activation = activation.getFunction();
        this.filters = filters;
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;
    }

    @Override
    public Tensor forward(ForwardContext context) {
        return null;
    }

    @Override
    public int size() {
        return filters;
    }

    public int filters() {
        return filters;
    }

    public ConvLayer filters(int filters) {
        this.filters = filters;
        return this;
    }

    public int kernelWidth() {
        return kernelWidth;
    }

    public int kernelHeight() {
        return kernelHeight;
    }

    public ConvLayer kernelSize(int kernelWidth, int kernelHeight) {
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;
        return this;
    }

    public int channels() {
        return channels;
    }

    public ConvLayer channels(int channels) {
        this.channels = channels;
        return this;
    }

    public int stride() {
        return stride;
    }

    public ConvLayer stride(int stride) {
        this.stride = stride;
        return this;
    }

    public int padding() {
        return padding;
    }

    public ConvLayer padding(int padding) {
        this.padding = padding;
        return this;
    }
}
