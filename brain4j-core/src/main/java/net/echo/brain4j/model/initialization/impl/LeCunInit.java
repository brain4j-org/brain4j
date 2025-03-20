package net.echo.brain4j.model.initialization.impl;

import net.echo.brain4j.model.initialization.WeightInitializer;

public class LeCunInit implements WeightInitializer {

    private static final double SQRT_OF_3 = Math.sqrt(3);

    @Override
    public double getBound(int nIn, int nOut) {
        return SQRT_OF_3 / Math.sqrt(nIn);
    }
}
