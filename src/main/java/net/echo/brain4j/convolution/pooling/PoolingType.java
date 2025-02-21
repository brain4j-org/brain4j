package net.echo.brain4j.convolution.pooling;

import net.echo.brain4j.convolution.pooling.impl.AveragePooling;
import net.echo.brain4j.convolution.pooling.impl.MaxPooling;

public enum PoolingType {

    MAX(new MaxPooling()),
    AVERAGE(new AveragePooling());

    private final PoolingFunction function;

    PoolingType(PoolingFunction function) {
        this.function = function;
    }

    public PoolingFunction getFunction() {
        return function;
    }
}
