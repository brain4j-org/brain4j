package net.echo.brain4j.initialization.impl;

import net.echo.brain4j.initialization.WeightInitializer;

public class NormalInit implements WeightInitializer {

    @Override
    public double getBound(int nIn, int nOut) {
        return 1;
    }
}
