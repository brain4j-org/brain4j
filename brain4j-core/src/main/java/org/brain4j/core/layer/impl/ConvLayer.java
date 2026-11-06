package org.brain4j.core.layer.impl;

import java.util.List;
import java.util.random.RandomGenerator;

import jdk.jfr.Experimental;

import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.Linear;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

/**
 * Applies 2D convolution with bias and activation.
 * Use it to extract spatial features from image-like inputs.
 */
@Experimental
public class ConvLayer extends Layer {

    /**
     * Holds filter count, kernel size, stride, padding and activation.
     */
    public record Config(int filters, int kernelWidth, int kernelHeight, int stride, int padding, Activation activation) {}

    protected Config config;
    private int channels;

    /**
     * Create a layer with stride 1 and linear activation.
     */
    public ConvLayer(int filters, int kernelWidth, int kernelHeight) {
        this(filters, kernelWidth, kernelHeight, new Linear());
    }

    /**
     * Create a layer with the given stride and linear activation.
     */
    public ConvLayer(int filters, int kernelWidth, int kernelHeight, int stride) {
        this(filters, kernelWidth, kernelHeight, stride, new Linear());
    }

    /**
     * Create a layer with stride 1 and the given activation.
     */
    public ConvLayer(int filters, int kernelWidth, int kernelHeight, Activation activation) {
        this(filters, kernelWidth, kernelHeight, 1, activation);
    }

    /**
     * Create a layer with the given stride and activation.
     */
    public ConvLayer(int filters, int kernelWidth, int kernelHeight, int stride, Activation activation) {
        this(filters, kernelWidth, kernelHeight, stride, 0, activation);
    }

    /**
     * Create a layer with the given stride, padding and activation.
     */
    public ConvLayer(int filters, int kernelWidth, int kernelHeight, int stride, int padding, Activation activation) {
        this(new Config(filters, kernelWidth, kernelHeight, stride, padding, activation));
    }

    /**
     * Create a layer with the given config.
     */
    public ConvLayer(Config config) {
        super(config.activation);

        if (config.stride <= 0) {
            throw Commons.illegalArgument("Stride must be > 0. Got: %s", config.stride);
        }

        if (config.filters <= 0) {
            throw Commons.illegalArgument("Filters must be > 0 Got: %s", config.filters);
        }

        if (config.kernelWidth <= 0) {
            throw Commons.illegalArgument("Kernel width must be > 0 Got: %s", config.kernelWidth);
        }

        if (config.kernelHeight <= 0) {
            throw Commons.illegalArgument("Kernel height must be > 0 Got: %s", config.kernelHeight);
        }

        this.config = config;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void build(List<Shape> inputShapes) {
        Shape inputShape = inputShapes.getFirst();
        this.channels = inputShape.dim(0);

        Tensor kernel = Tensors.zeros(config.filters, channels, config.kernelHeight, config.kernelWidth);
        Tensor bias = Tensors.zeros(config.filters);

        registerParam("kernel", kernel);
        registerParam("bias", bias);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
        int input = channels * config.kernelHeight * config.kernelWidth;
        int output = config.filters * config.kernelHeight * config.kernelWidth;

        generateWeights("kernel", rng, input, output);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<Shape> inferOutputShapes(List<Shape> inputShapes) {
        if (inputShapes.size() != 1) {
            throw Commons.illegalArgument("Layer requires 1 input but %s were given!", inputShapes.size());
        }

        Shape inputShape = inputShapes.getFirst();

        if (inputShape.rank() != 3) {
            throw Commons.illegalArgument("Conv requires tensors with rank 3 but %s were given!", inputShape.rank());
        }

        int height = inputShape.dim(1);
        int width = inputShape.dim(2);

        int numeratorH = height - config.kernelHeight;
        int numeratorW = width - config.kernelWidth;

        if (numeratorH < 0 || numeratorW < 0) {
            throw Commons.illegalArgument("Kernel is too big for input!.");
        }

        int outHeight = numeratorH / config.stride + 1;
        int outWidth = numeratorW / config.stride + 1;

        if (outHeight <= 0 || outWidth <= 0) {
            throw Commons.illegalArgument("Negative output dims: outHeight=%s outWidth=%s", outHeight, outWidth);
        }

        return List.of(Shape.of(config.filters, outHeight, outWidth));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];

        if (input.rank() != 4) {
            throw Commons.illegalArgument("Expected input with rank 4 but got %s", input.rank());
        }

        Tensor w = param("kernel");
        Tensor b = param("bias");

        Tensor result = input.convolve(w, config.stride)
            .add(b.reshape(1, config.filters, 1, 1))
            .activate(activation);

        return tensors(result);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Layer copy() {
        ConvLayer copy = new ConvLayer(config);
        copyParameters(copy);

        return copy;
    }

    /**
     * Return the input channel count.
     *
     * @return the channel count
     */
    public int channels() {
        return channels;
    }

    /**
     * Return the layer config.
     *
     * @return the config
     */
    public Config config() {
        return config;
    }
}
