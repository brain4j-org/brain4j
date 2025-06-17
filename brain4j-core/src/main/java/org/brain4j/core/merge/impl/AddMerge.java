package org.brain4j.core.merge.impl;

import org.brain4j.core.merge.MergeStrategy;
import org.brain4j.math.tensor.Tensor;

public class AddMerge implements MergeStrategy {

    @Override
    public Tensor process(Tensor... inputs) {
        if (inputs.length < 2) {
            throw new IllegalArgumentException("Merging requires at least two input tensors.");
        }

        Tensor result = inputs[0];

        for (int i = 1; i < inputs.length; i++) {
            result = result.addGrad(inputs[i]);
        }

        return result;
    }
}
