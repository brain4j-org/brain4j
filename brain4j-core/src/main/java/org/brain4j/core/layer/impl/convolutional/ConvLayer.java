package org.brain4j.core.layer.impl.convolutional;

import org.brain4j.core.activation.Activations;
import org.brain4j.core.layer.ForwardContext;
import org.brain4j.core.layer.Layer;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.util.Random;

public class ConvLayer extends Layer {

    private int filters;
    private int kernelWidth;
    private int kernelHeight;
    private int channels;
    private int stride = 1;
    private int padding = 0;

    public ConvLayer(Activations activation, int filters, int kernelWidth, int kernelHeight) {
        this.activation = activation.function();
        this.filters = filters;
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;
    }

    @Override
    public Layer connect(Layer previous) {
        channels = 1;

        if (previous != null) {
            channels = previous.size();
        }

        this.bias = Tensors.zeros(filters);
        this.weights = Tensors.zeros(filters, channels, kernelHeight, kernelWidth);

        return this;
    }

    @Override
    public void initWeights(Random generator, int input, int output) {
        this.bias.map(x -> weightInit.generate(generator, input, output));
        this.weights.map(x -> weightInit.generate(generator, input, output));
    }

    @Override
    public Tensor forward(ForwardContext context) {
        Tensor input = context.input();
        int[] shape = input.shape();

        if (shape[1] != channels) {
            throw new IllegalArgumentException("Input channel mismatch: " + shape[1] + " != " + channels);
        }

        return input.convolve(weights, stride, padding);
    }

    @Override
    public int size() {
        return filters;
    }

    @Override
    public boolean validateInput(Tensor input) {
        // [batch_size, channels, height, width]
        return input.shape()[2] == channels;
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
