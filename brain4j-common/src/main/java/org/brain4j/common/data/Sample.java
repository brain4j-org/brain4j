package org.brain4j.common.data;

import org.brain4j.common.tensor.Tensor;

import java.util.Arrays;

public class Sample {
    private final Tensor[] inputs;
    private final Tensor label;

    public Sample(Tensor input, Tensor label) {
        this(new Tensor[]{input}, label);
    }

    public Sample(Tensor[] inputs, Tensor label) {
        this.inputs = inputs;
        this.label = label;
    }

    public Tensor[] inputs() {
        return inputs;
    }

    public Tensor input() {
        return inputs[0];
    }

    public Tensor label() {
        return label;
    }

    @Override
    public String toString() {
        if (inputs.length == 1) {
            return inputs[0].toString("%.3f") + " -> " + label.toString("%.3f");
        }

        return Arrays.toString(inputs) + " -> " + label.toString("%.3f");
    }
}
