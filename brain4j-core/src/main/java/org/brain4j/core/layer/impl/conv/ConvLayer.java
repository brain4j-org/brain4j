package org.brain4j.core.layer.impl.conv;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.structure.StatesCache;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.io.DataInputStream;
import java.io.DataOutputStream;

public class ConvLayer extends Layer {

    private Tensor filters;
    private Tensor bias;

    private int channels;
    private int filtersWidth;
    private int filtersHeight;
    private int padding;
    private int stride;

    public ConvLayer(Activations activation, int filters, int filtersWidth, int filtersHeight) {
        this(activation, filters, 1, filtersWidth, filtersHeight);
    }

    public ConvLayer(Activations activation, int filters, int channels, int filtersWidth, int filtersHeight) {
        this(activation.getFunction(), filters, channels, filtersWidth, filtersHeight, 0, 1);
    }

    public ConvLayer(Activation activation, int filters, int channels, int filtersWidth, int filtersHeight, int padding, int stride) {
        super(activation);
        this.filters = Tensors.create(filters, channels, filtersHeight, filtersWidth);
        this.bias = Tensors.create(channels, filtersHeight, filtersWidth);
        this.filtersWidth = filtersWidth;
        this.filtersHeight = filtersHeight;
        this.padding = padding;
        this.stride = stride;
    }

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        super.serialize(stream);
        filters.serialize(stream);
        bias.serialize(stream);
        stream.writeInt(channels);
        stream.writeInt(filtersWidth);
        stream.writeInt(filtersHeight);
        stream.writeInt(padding);
        stream.writeInt(stride);
    }

    @Override
    public void deserialize(DataInputStream stream) throws Exception {
        super.deserialize(stream);
        this.filters = Tensors.zeros(0).deserialize(stream);
        this.bias = Tensors.zeros(0).deserialize(stream);
        this.channels = stream.readInt();
        this.filtersWidth = stream.readInt();
        this.filtersHeight = stream.readInt();
        this.padding = stream.readInt();
        this.stride = stream.readInt();
    }

    @Override
    public Tensor forward(StatesCache cache, Layer lastLayer, Tensor input, boolean training) {
        return input.convolve(filters).map(x -> activation.activate(x));
    }

    public Tensor getFilters() {
        return filters;
    }

    public int getChannels() {
        return channels;
    }

    public int getFiltersWidth() {
        return filtersWidth;
    }

    public int getFiltersHeight() {
        return filtersHeight;
    }

    public int getPadding() {
        return padding;
    }

    public int getStride() {
        return stride;
    }
}
