package org.brain4j.core.layer.impl.convolutional;

import org.brain4j.core.clipper.impl.HardClipper;
import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;
import org.brain4j.math.activation.impl.LinearActivation;
import org.brain4j.math.tensor.Tensor;

public class MaxPoolingLayer extends Layer {

    private int kernelWidth;
    private int kernelHeight;
    private int stride;
    private int channels;
    private int outputSize;

    public MaxPoolingLayer(int kernelWidth, int kernelHeight) {
        this(kernelWidth, kernelHeight, 1);
    }

    public MaxPoolingLayer(int kernelWidth, int kernelHeight, int stride) {
        super(new LinearActivation(), new HardClipper(5));
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;
        this.stride = stride;
    }

    @Override
    public void connect(Layer previous, Layer next) {
        Tensor weights = previous.weights();
        this.channels = weights.shape()[0]; // [filters, channels, height, width]
    }

    @Override
    public Tensor forward(ForwardContext context) {
        return null;
    }

    @Override
    public int size() {
        return channels;
    }
}
