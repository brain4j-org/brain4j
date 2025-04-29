package net.echo.math.data;

import net.echo.math.tensor.Tensor;

public record Sample(Tensor input, Tensor label) {

    @Override
    public String toString() {
        return input.toString("%.3f") + " -> " + label.toString("%.3f");
    }
}
