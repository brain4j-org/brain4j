package net.echo.brain4j.layer.impl.conv;

import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.StatesCache;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ConvLayer extends Layer {

    private List<Tensor> kernels;
    private int filters;
    private int kernelWidth;
    private int kernelHeight;
    private int stride;
    private int padding;
    private int outputWidth;
    private int outputHeight;

    private ConvLayer() {
    }
    
    public ConvLayer(int filters, int kernelWidth, int kernelHeight) {
        this(filters, kernelWidth, kernelHeight, 1, 0);
    }

    public ConvLayer(int filters, int kernelWidth, int kernelHeight, int stride) {
        this(filters, kernelWidth, kernelHeight, stride, 0);
    }

    public ConvLayer(int filters, int kernelWidth, int kernelHeight, int stride, int padding) {
        this.kernels = new ArrayList<>();
        this.filters = filters;
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;
        this.stride = stride;
        this.padding = padding;
    }

    @Override
    public boolean isConvolutional() {
        return true;
    }

    @Override
    public int getTotalNeurons() {
        return filters;
    }

    @Override
    public int getTotalParams() {
        return filters * kernels.getFirst().elements();
    }

    @Override
    public void connect(Random generator, Layer previous, Layer next, double bound) {
        int channels = 1;

        if (previous instanceof InputLayer inputLayer) {
            this.outputWidth = (inputLayer.getWidth() - kernelWidth + 2 * padding) / stride + 1;
            this.outputHeight = (inputLayer.getHeight() - kernelHeight + 2 * padding) / stride + 1;
        }

        if (previous instanceof ConvLayer convLayer) {
            channels = convLayer.getFilters();

            this.outputWidth = (convLayer.getOutputWidth() - kernelWidth + 2 * padding) / stride + 1;
            this.outputHeight = (convLayer.getOutputHeight() - kernelHeight + 2 * padding) / stride + 1;
        }

        for (int i = 0; i < filters; i++) {
            Tensor kernel = TensorFactory.create(channels, kernelHeight, kernelWidth);

            for (int j = 0; j < kernel.elements(); j++) {
                double value = 2 * generator.nextDouble() * bound - bound;
                kernel.getData().set(j, value);
            }

            kernels.add(kernel);
        }
    }

    @Override
    public Tensor forward(StatesCache cache, Layer lastLayer, Tensor input, boolean training) {
        cache.setInputTensor(this, input);

        int channels = input.shape()[0];
        int columns = input.shape()[1];
        int rows = input.shape()[2];

        int outputWidth = (columns - kernelWidth + 2 * padding) / stride + 1;
        int outputHeight = (rows - kernelHeight + 2 * padding) / stride + 1;

        Tensor result = TensorFactory.matrix(filters, outputHeight, outputWidth);

        for (int f = 0; f < filters; f++) {
            Tensor kernel = kernels.get(f);
            Tensor filterOutput = TensorFactory.matrix(outputHeight, outputWidth);

            for (int c = 0; c < channels; c++) {
                Tensor inputSlice = input.slice(c);
                Tensor kernelSlice = kernel.slice(c);

                Tensor convResult = inputSlice.convolve(kernelSlice); // TODO: add stride and padding
                filterOutput.add(convResult);
            }

            result.setChannel(f, filterOutput);
        }

        cache.setOutputTensor(this, result);
        return result;
    }

    public int getOutputWidth() {
        return outputWidth;
    }

    public int getOutputHeight() {
        return outputHeight;
    }

    public List<Tensor> getKernels() {
        return kernels;
    }

    public int getFilters() {
        return filters;
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
