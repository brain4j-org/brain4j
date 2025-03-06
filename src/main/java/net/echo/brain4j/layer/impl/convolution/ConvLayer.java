package net.echo.brain4j.layer.impl.convolution;

import com.google.common.base.Preconditions;
import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.convolution.Kernel;
import net.echo.brain4j.layer.Layer;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.structure.Neuron;
import net.echo.brain4j.structure.cache.Parameters;
import net.echo.brain4j.structure.cache.StatesCache;
import net.echo.brain4j.training.optimizers.Optimizer;
import net.echo.brain4j.training.updater.Updater;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static net.echo.brain4j.utils.MLUtils.clipGradient;

/**
 * Represents a convolutional layer used for feature extraction and image classification.
 */
public class ConvLayer extends Layer<Kernel, Kernel> {

    protected final List<Kernel> kernels = new ArrayList<>();

    protected final int kernelWidth;
    protected final int kernelHeight;
    protected final int filters;

    protected int padding;
    protected int stride;

    /**
     * Constructs a convolutional layer instance with the given parameters.
     *
     * @param filters the amount of filters, more filters means more features
     * @param kernelWidth the width of each filter
     * @param kernelHeight the height of each filter
     * @param activation the activation function
     */
    public ConvLayer(int filters, int kernelWidth, int kernelHeight, Activations activation) {
        this(filters, kernelWidth, kernelHeight, 1, 0, activation);
    }

    /**
     * Constructs a convolutional layer instance with the given parameters.
     *
     * @param filters the amount of filters, more filters means more features
     * @param kernelWidth the width of each filter
     * @param kernelHeight the height of each filter
     * @param stride the stride to apply to each convolution
     * @param activation the activation function
     */
    public ConvLayer(int filters, int kernelWidth, int kernelHeight, int stride, Activations activation) {
        this(filters, kernelWidth, kernelHeight, stride, 0, activation);
    }

    /**
     * Constructs a convolutional layer instance with the given parameters.
     *
     * @param filters the amount of filters, more filters means more features
     * @param kernelWidth the width of each filter
     * @param kernelHeight the height of each filter
     * @param stride the stride to apply to each convolution
     * @param padding the padding to apply to each convolution
     * @param activation the activation function
     */
    public ConvLayer(int filters, int kernelWidth, int kernelHeight, int stride, int padding, Activations activation) {
        super(0, activation);
        this.id = Parameters.TOTAL_CONV_LAYER++;
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
            Kernel kernel = new Kernel(i, kernelWidth, kernelHeight);
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

        result.apply(activation.getFunction());
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

        Kernel kernel = postProcess(featureMap).padding(padding);

        cache.setInput(this, input);
        cache.setFeatureMap(this, kernel);

        return kernel;
    }

    @Override
    public void propagate(StatesCache cache, Layer<?, ?> nextLayer, Updater updater, Optimizer optimizer) {
        Kernel featureMap = cache.getFeatureMap(this);

        if (nextLayer instanceof FlattenLayer) {
            List<Neuron> neurons = nextLayer.getNeurons();
            Kernel deltaKernel = new Kernel(featureMap.getWidth(), featureMap.getHeight());

            for (int h = 0; h < featureMap.getHeight(); h++) {
                for (int w = 0; w < featureMap.getWidth(); w++) {
                    int index = h * featureMap.getWidth() + w;
                    double deltaNeuron = neurons.get(index).getDelta(cache);

                    double derivative = activation.getFunction().getDerivative(featureMap.getValue(w, h));
                    double localDelta = deltaNeuron * derivative;

                    deltaKernel.setValue(w, h, localDelta);
                }
            }

            updateParameters(cache, optimizer, deltaKernel);
        } else if (nextLayer instanceof ConvLayer nextConvLayer) {
            Kernel deltaNext = cache.getDelta(nextConvLayer);
            Kernel deltaCurrent = new Kernel(featureMap.getWidth(), featureMap.getHeight());

            for (Kernel nextKernel : nextConvLayer.getKernels()) {
                Kernel rotatedKernel = nextKernel.rotate180();
                Kernel contribution = deltaNext.convolute(rotatedKernel, 1);

                if (contribution.getWidth() != deltaCurrent.getWidth() || contribution.getHeight() != deltaCurrent.getHeight()) {
                    contribution = cropTo(contribution, deltaCurrent.getWidth(), deltaCurrent.getHeight());
                }

                deltaCurrent.add(contribution);
            }

            for (int h = 0; h < deltaCurrent.getHeight(); h++) {
                for (int w = 0; w < deltaCurrent.getWidth(); w++) {
                    double derivative = activation.getFunction().getDerivative(featureMap.getValue(w, h));
                    double updatedDelta = clipGradient(deltaCurrent.getValue(w, h) * derivative);

                    deltaCurrent.setValue(w, h, updatedDelta);
                }
            }

            updateParameters(cache, optimizer, deltaCurrent);
        } else {
            throw new UnsupportedOperationException("Propagation not support for " + nextLayer.getClass().getSimpleName());
        }
    }

    private void updateParameters(StatesCache cache, Optimizer optimizer, Kernel deltaKernel) {
        cache.setDelta(this, deltaKernel);
        Kernel input = cache.getInput(this);

        for (Kernel kernel : kernels) {
            Kernel grad = input.convolute(deltaKernel, stride);

            for (int h = 0; h < kernel.getHeight(); h++) {
                for (int w = 0; w < kernel.getWidth(); w++) {
                    float weight = kernel.getValue(w, h);
                    float gradient = clipGradient(grad.getValue(w, h));

                    float update = (float) optimizer.update(cache, kernel.getId(), gradient, weight);
                    kernel.setValue(w, h, weight - update);
                }
            }
        }
    }

    private Kernel cropTo(Kernel source, int targetWidth, int targetHeight) {
        int sourceWidth = source.getWidth();
        int sourceHeight = source.getHeight();

        int offsetW = (targetWidth - sourceWidth) / 2;
        int offsetH = (targetHeight - sourceHeight) / 2;

        Kernel cropped = new Kernel(targetWidth, targetHeight);

        for (int h = 0; h < targetHeight; h++) {
            for (int w = 0; w < targetWidth; w++) {
                cropped.setValue(w, h, source.getValue(w / offsetW, h / offsetH));
            }
        }

        return cropped;
    }

    @Override
    public int getTotalParams() {
        return filters * kernelWidth * kernelHeight;
    }

    @Override
    public int getTotalNeurons() {
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
