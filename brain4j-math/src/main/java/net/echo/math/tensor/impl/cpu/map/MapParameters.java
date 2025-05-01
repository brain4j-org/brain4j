package net.echo.math.tensor.impl.cpu.map;

import net.echo.math.lang.DoubleToDoubleFunction;

public record MapParameters(
        DoubleToDoubleFunction function,
        float[] data
) {

}
