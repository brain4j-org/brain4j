package org.brain4j.core.layer.impl;

import jdk.jfr.Experimental;
import org.brain4j.core.layer.Layer;
import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.impl.Linear;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.data.StatesCache;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

import java.util.List;
import java.util.random.RandomGenerator;

@Experimental
public class ConvLayer extends Layer {

    public record Config(int filters, int kernelWidth, int kernelHeight, int stride, int padding, Activation activation) {}

    protected Config config;
    private int channels;

    public ConvLayer(int filters, int kernelWidth, int kernelHeight) {
        this(filters, kernelWidth, kernelHeight, new Linear());
    }

    public ConvLayer(int filters, int kernelWidth, int kernelHeight, int stride) {
        this(filters, kernelWidth, kernelHeight, stride, new Linear());
    }

    public ConvLayer(int filters, int kernelWidth, int kernelHeight, Activation activation) {
        this(filters, kernelWidth, kernelHeight, 1, activation);
    }

    public ConvLayer(int filters, int kernelWidth, int kernelHeight, int stride, Activation activation) {
        this(filters, kernelWidth, kernelHeight, stride, 0, activation);
    }

    public ConvLayer(int filters, int kernelWidth, int kernelHeight, int stride, int padding, Activation activation) {
        this(new Config(filters, kernelWidth, kernelHeight, stride, padding, activation));
    }

    public ConvLayer(Config config) {
        super(config.activation);

        if (config.stride <= 0) throw Commons.illegalArgument("Stride must be > 0. Got: %s", config.stride);
        if (config.filters <= 0) throw Commons.illegalArgument("Filters must be > 0 Got: %s", config.filters);
        if (config.kernelWidth <= 0) throw Commons.illegalArgument("Kernel width must be > 0 Got: %s", config.kernelWidth);
        if (config.kernelHeight <= 0) throw Commons.illegalArgument("Kernel height must be > 0 Got: %s", config.kernelHeight);

        this.config = config;
    }

    @Override
    public void build(List<Shape> inputShapes) {
        Shape inputShape = inputShapes.getFirst();
        this.channels = inputShape.dim(0);

        Tensor kernel = Tensors.zeros(config.filters, channels, config.kernelHeight, config.kernelWidth);
        Tensor bias = Tensors.zeros(config.filters);

        registerParam("kernel", kernel);
        registerParam("bias", bias);
    }

    @Override
    public void initWeights(List<Shape> inputShapes, RandomGenerator rng) {
        int input = channels * config.kernelHeight * config.kernelWidth;
        int output = config.filters * config.kernelHeight * config.kernelWidth;

        generateWeights("kernel", rng, input, output);
    }

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

        int numeratorH = height - config.kernelHeight; // + 2 * padding;
        int numeratorW = width - config.kernelWidth; // + 2 * padding;

        if (numeratorH < 0 || numeratorW < 0) {
            throw Commons.illegalArgument("Kernel is too big for input!.");
        }

        int outHeight = numeratorH / config.stride + 1;
        int outWidth  = numeratorW / config.stride + 1;

        if (outHeight <= 0 || outWidth <= 0) {
            throw Commons.illegalArgument("Negative output dims: outHeight=%s outWidth=%s", outHeight, outWidth);
        }

        return List.of(Shape.of(config.filters, outHeight, outWidth));
    }

    @Override
    public Tensor[] forward(StatesCache cache, Tensor... inputs) {
        Tensor input = inputs[0];

        if (input.rank() != 4) {
            throw Commons.illegalArgument("Expected input with rank 4 but got %s", input.rank());
        }

        Tensor W = getParam("kernel");
        Tensor B = getParam("bias");

        Tensor result = input.convolveGrad(W, config.stride)
            .addGrad(B.reshapeGrad(1, config.filters, 1, 1))
            .activateGrad(activation);

        return tensors(result);
    }

    @Override
    public Layer copy() {
        ConvLayer copy = new ConvLayer(config);
        copyParameters(copy);
        return copy;
    }

    public int channels() {
        return channels;
    }

    public Config config() {
        return config;
    }
}
