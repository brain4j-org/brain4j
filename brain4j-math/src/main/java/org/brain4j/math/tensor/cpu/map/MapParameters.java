package org.brain4j.math.tensor.cpu.map;

import org.brain4j.math.lang.DoubleToDoubleFunction;

public record MapParameters(
        DoubleToDoubleFunction function,
        float[] data
) {

}
