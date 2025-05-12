package org.brain4j.core.layer.impl.conv;

import org.brain4j.core.layer.Layer;
import org.brain4j.core.structure.StatesCache;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.util.Random;

public class ConvLayer extends Layer {

    private int filters;
    private int channels;
    private int filtersWidth;
    private int filtersHeight;
    private int padding;
    private int stride;

    public ConvLayer(Activations activation, int filters, int filtersWidth, int filtersHeight) {
        this(activation.getFunction(), filters, filtersWidth, filtersHeight, 1, 1);
    }

    public ConvLayer(Activation activation, int filters, int filtersWidth, int filtersHeight, int padding, int stride) {
        this.id = totalLayers++;
        this.activation = activation;
        this.filters = filters;
        this.filtersWidth = filtersWidth;
        this.filtersHeight = filtersHeight;
        this.padding = padding;
        this.stride = stride;
        this.weights = Tensors.create(filters);
    }

    @Override
    public void connect(Random generator, Layer previous, double bound) {
        if (previous instanceof ConvLayer convLayer) {
            this.channels = convLayer.getChannels();
        } else if (previous instanceof InputLayer inputLayer) {
            this.channels = inputLayer.getChannels();
        } else {
            throw new IllegalArgumentException("Convolutional layer is not preceded by anything!");
        }

        this.bias = Tensors.matrix(channels, filtersHeight, filtersWidth);

        for (int i = 0; i < bias.elements(); i++) {
            bias.getData()[i] = (float) (2 * generator.nextDouble() - 1);
        }

        for (int i = 0; i < weights.elements(); i++) {
            weights.getData()[i] = (float) (2 * generator.nextDouble() - 1);
        }
    }

    @Override
    public void serialize(DataOutputStream stream) throws Exception {
        super.serialize(stream);
        stream.writeInt(filters);
        stream.writeInt(channels);
        stream.writeInt(filtersWidth);
        stream.writeInt(filtersHeight);
        stream.writeInt(padding);
        stream.writeInt(stride);
    }

    @Override
    public void deserialize(DataInputStream stream) throws Exception {
        super.deserialize(stream);
        this.filters = stream.readInt();
        this.channels = stream.readInt();
        this.filtersWidth = stream.readInt();
        this.filtersHeight = stream.readInt();
        this.padding = stream.readInt();
        this.stride = stream.readInt();
    }

    @Override
    public Tensor forward(StatesCache cache, Tensor input, boolean training) {
        // [batch_size, channels, height, width]
        return input.convolve(weights)
                .map(x -> activation.activate(x));
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
