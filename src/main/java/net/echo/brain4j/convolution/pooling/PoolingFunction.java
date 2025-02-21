package net.echo.brain4j.convolution.pooling;

import net.echo.brain4j.convolution.Kernel;
import net.echo.brain4j.convolution.impl.PoolingLayer;

public interface PoolingFunction {

    double apply(PoolingLayer layer, Kernel input, int i, int j);
}
