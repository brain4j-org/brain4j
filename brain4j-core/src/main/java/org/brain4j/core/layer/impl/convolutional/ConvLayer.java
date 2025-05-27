package org.brain4j.core.layer.impl.convolutional;

import org.brain4j.core.clipper.GradientClipper;
import org.brain4j.core.clipper.impl.HardClipper;
import org.brain4j.core.layer.Layer;
import org.brain4j.core.training.StatesCache;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.weights.WeightInitialization;

public class ConvLayer extends Layer {

    private int filters;
    private int kernelWidth;
    private int kernelHeight;
    private int channels;
    private int stride;
    private int padding;

    public ConvLayer(Activations activation, int filters, int kernelWidth, int kernelHeight) {
        this(activation, filters, kernelWidth, kernelHeight, 1);
    }

    public ConvLayer(Activations activation, int filters, int kernelWidth, int kernelHeight, int stride) {
        this(activation, filters, kernelWidth, kernelHeight, stride, 0);
    }


    public ConvLayer(Activations activation, int filters, int kernelWidth, int kernelHeight, int stride, int padding) {
        this(activation.getFunction(), new HardClipper(5), activation.getFunction().defaultWeightInit(), filters,
            kernelWidth, kernelHeight, stride, padding);
    }

    public ConvLayer(Activation activation, GradientClipper clipper, WeightInitialization weightInit, int filters,
                     int kernelWidth, int kernelHeight, int stride, int padding) {
        super(activation, clipper, weightInit);
        this.filters = filters;
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;
        this.stride = stride;
        this.padding = padding;
    }

    @Override
    public Tensor forward(StatesCache cache, Tensor input, int index, boolean training) {
        return null;
    }

    @Override
    public int size() {
        return filters;
    }
}
