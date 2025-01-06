package net.echo.brain4j.model.initialization.impl;

import net.echo.brain4j.model.initialization.WeightInitializer;

public class UniformXavierInit implements WeightInitializer {

    @Override
    public double getBound(int nIn, int nOut) {
        return Math.sqrt(6.0 / (nIn + nOut));
    }
}
