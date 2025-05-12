package org.brain4j.math.data;

import org.brain4j.math.tensor.Tensor;

public record Sample(Tensor input, Tensor label) {

    @Override
    public String toString() {
        return input.toString("%.3f") + " -> " + label.toString("%.3f");
    }
}
