package net.echo.brain4j.convolution.pooling;

import net.echo.brain4j.convolution.Kernel;
import net.echo.brain4j.layer.impl.convolution.PoolingLayer;

public interface PoolingFunction {

    double apply(PoolingLayer layer, Kernel input, int i, int j);

    void unpool(PoolingLayer layer, int outX, int outY, Kernel deltaPooling, Kernel deltaUnpooled, Kernel input);
}
