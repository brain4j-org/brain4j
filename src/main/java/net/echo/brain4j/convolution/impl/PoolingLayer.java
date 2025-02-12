package net.echo.brain4j.convolution.impl;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.convolution.Kernel;
import net.echo.brain4j.convolution.pooling.PoolingType;
import net.echo.brain4j.layer.Layer;

public class PoolingLayer extends Layer {

    protected final PoolingType poolingType;
    protected final int kernelWidth;
    protected final int kernelHeight;
    protected int stride;
    protected int padding;

    public PoolingLayer(PoolingType poolingType, int kernelWidth, int kernelHeight) {
        this(poolingType, kernelWidth, kernelHeight, 1, 0);
    }

    public PoolingLayer(PoolingType poolingType, int kernelWidth, int kernelHeight, int stride) {
        this(poolingType, kernelWidth, kernelHeight, stride, 0);
    }

    public PoolingLayer(PoolingType poolingType, int kernelWidth, int kernelHeight, int stride, int padding) {
        super(kernelWidth * kernelHeight, Activations.LINEAR);
        this.poolingType = poolingType;
        this.kernelHeight = kernelHeight;
        this.kernelWidth = kernelWidth;
        this.stride = stride;
        this.padding = padding;
    }

    @Override
    public boolean isConvolutional() {
        return true;
    }

    public Kernel applcyPooling(Kernel input) {
        int outputWidth = (input.getWidth() - kernelWidth + 2 * padding) / stride + 1;
        int outputHeight = (input.getHeight() - kernelHeight + 2 * padding) / stride + 1;

        Kernel output = new Kernel(outputWidth, outputHeight);

        // Applicazione del pooling
        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < outputWidth; j++) {
                double pooledValue;
                if (poolingType == PoolingType.MAX) {
                    pooledValue = Double.NEGATIVE_INFINITY;

                    for (int ki = 0; ki < kernelHeight; ki++) {
                        for (int kj = 0; kj < kernelWidth; kj++) {
                            int x = j * stride + kj - padding;
                            int y = i * stride + ki - padding;

                            if (x >= 0 && x < input.getWidth() && y >= 0 && y < input.getHeight()) {
                                double value = input.getValue(x, y);
                                pooledValue = Math.max(pooledValue, value);
                            }
                        }
                    }
                } else if (poolingType == PoolingType.AVERAGE) {
                    double sum = 0;
                    int count = 0;

                    for (int ki = 0; ki < kernelHeight; ki++) {
                        for (int kj = 0; kj < kernelWidth; kj++) {
                            int x = j * stride + kj - padding;
                            int y = i * stride + ki - padding;

                            if (x >= 0 && x < input.getWidth() && y >= 0 && y < input.getHeight()) {
                                sum += input.getValue(x, y);
                                count++;
                            }
                        }
                    }
                    pooledValue = sum / count;
                } else {
                    throw new UnsupportedOperationException("Unknown pooling type: " + poolingType);
                }

                output.setValue(j, i, pooledValue);
            }
        }

        return output;
    }
}
