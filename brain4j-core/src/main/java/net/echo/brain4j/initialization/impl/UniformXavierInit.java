package net.echo.brain4j.initialization.impl;

import net.echo.brain4j.initialization.WeightInitializer;

public class UniformXavierInit implements WeightInitializer {

    @Override
    public double getBound(int nIn, int nOut) {
        return Math.sqrt(6.0 / (nIn + nOut));
    }
}
