package net.echo.brain4j.convolution.pooling.impl;

import net.echo.brain4j.convolution.Kernel;
import net.echo.brain4j.convolution.pooling.PoolingFunction;
import net.echo.brain4j.layer.impl.convolution.PoolingLayer;

public class AveragePooling implements PoolingFunction {

    @Override
    public double apply(PoolingLayer layer, Kernel input, int i, int j) {
        double sum = 0;
        int count = 0;

        int stride = layer.getStride();
        int padding = layer.getPadding();

        int startX = j * stride - padding;
        int startY = i * stride - padding;

        for (int ki = 0; ki < layer.getKernelHeight(); ki++) {
            for (int kj = 0; kj < layer.getKernelWidth(); kj++) {
                int x = startX + kj;
                int y = startY + ki;

                if (x >= 0 && x < input.getWidth() && y >= 0 && y < input.getHeight()) {
                    sum += input.getValue(x, y);
                    count++;
                }
            }
        }

        return sum / count;
    }

    @Override
    public void unpool(PoolingLayer layer, int outX, int outY, Kernel deltaPooling, Kernel deltaUnpooled, Kernel input) {
        double deltaVal = deltaPooling.getValue(outX, outY);

        int startX = outX * layer.getStride();
        int startY = outY * layer.getStride();

        int endX = Math.min(startX + layer.getKernelWidth(), input.getWidth());
        int endY = Math.min(startY + layer.getKernelHeight(), input.getHeight());

        int poolArea = (endX - startX) * (endY - startY);
        double distributedDelta = deltaVal / poolArea;

        for (int y = startY; y < endY; y++) {
            for (int x = startX; x < endX; x++) {
                double current = deltaUnpooled.getValue(x, y);
                deltaUnpooled.setValue(x, y, current + distributedDelta);
            }
        }
    }
}
