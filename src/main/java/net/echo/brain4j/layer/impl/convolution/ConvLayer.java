package net.echo.brain4j.layer.impl.convolution;

import com.google.common.base.Preconditions;
import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.convolution.Kernel;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ConvLayer extends Layer<Kernel, Kernel> {

    protected final List<Kernel> kernels = new ArrayList<>();

    protected final int kernelWidth;
    protected final int kernelHeight;
    protected final int filters;

    protected int padding;
    protected int stride;

    public ConvLayer(int filters, int kernelWidth, int kernelHeight, Activations activation) {
        this(filters, kernelWidth, kernelHeight, 1, 0, activation);
    }

    public ConvLayer(int filters, int kernelWidth, int kernelHeight, int stride, Activations activation) {
        this(filters, kernelWidth, kernelHeight, stride, 0, activation);
    }

    public ConvLayer(int filters, int kernelWidth, int kernelHeight, int stride, int padding, Activations activation) {
        super(0, activation);
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
    public void connectAll(Random generator, Layer<?, ?> nextLayer, double bound) {
        for (int i = 0; i < filters; i++) {
            Kernel kernel = new Kernel(kernelWidth, kernelHeight);
            kernel.setValues(generator, bound);

            this.kernels.add(kernel);
        }
    }

    public Kernel postProcess(List<Kernel> featureMap) {
        Preconditions.checkArgument(featureMap.size() == filters, "Feature map size does not match the number of filters!");

        Kernel first = featureMap.getFirst();
        Kernel result = new Kernel(first.getWidth(), first.getHeight());

        for (Kernel feature : featureMap) {
            result.add(feature);
        }

        result.apply(this.activation.getFunction());
        return result;
    }

    @Override
    public Kernel forward(StatesCache cache, Layer<?, ?> lastLayer, Kernel input) {
        Preconditions.checkNotNull(input, "Last convolutional input is null! Missing an input layer?");

        List<Kernel> featureMap = new ArrayList<>();

        for (Kernel kernel : kernels) {
            Kernel result = input.convolute(kernel, stride);

            featureMap.add(result);
        }

        return postProcess(featureMap).padding(padding);
    }

    @Override
    public void propagate(StatesCache cacheHolder, Layer<?, ?> nextLayer, Updater updater, Optimizer optimizer) {
        Kernel inputKernel = cacheHolder.getInputKernel(this);
        Kernel outputKernel = cacheHolder.getOutputKernel(this);
        Kernel errorMap = cacheHolder.getDeltaKernel(this);

        throw new UnsupportedOperationException("Not implemented yet.");
    }

    @Override
    public int getTotalParams() {
        return filters * kernelWidth * kernelHeight;
    }

    @Override
    public int size() {
        return filters;
    }

    public List<Kernel> getKernels() {
        return kernels;
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
