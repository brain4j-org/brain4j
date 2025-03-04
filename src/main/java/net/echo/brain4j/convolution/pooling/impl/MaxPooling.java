package net.echo.brain4j.convolution.pooling.impl;

import net.echo.brain4j.convolution.Kernel;
import net.echo.brain4j.layer.impl.convolution.PoolingLayer;
import net.echo.brain4j.convolution.pooling.PoolingFunction;

public class MaxPooling implements PoolingFunction {

    @Override
    public double apply(PoolingLayer layer, Kernel input, int i, int j) {
        double pooledValue = Double.NEGATIVE_INFINITY;

        int stride = layer.getStride();
        int padding = layer.getPadding();

        int startX = j * stride - padding;
        int startY = i * stride - padding;

        for (int ki = 0; ki < layer.getKernelHeight(); ki++) {
            for (int kj = 0; kj < layer.getKernelWidth(); kj++) {
                int x = startX + kj;
                int y = startY + ki;

                if (x >= 0 && x < input.getWidth() && y >= 0 && y < input.getHeight()) {
                    double value = input.getValue(x, y);
                    pooledValue = Math.max(pooledValue, value);
                }
            }
        }

        return pooledValue;
    }
}
