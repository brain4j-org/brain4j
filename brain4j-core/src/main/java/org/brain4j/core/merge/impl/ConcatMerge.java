package org.brain4j.core.merge.impl;

import org.brain4j.common.tensor.Tensor;
import org.brain4j.core.merge.MergeStrategy;

public class ConcatMerge implements MergeStrategy {

    @Override
    public Tensor process(Tensor... inputs) {
        if (inputs.length < 2) {
            throw new IllegalArgumentException("Merging requires at least two input tensors.");
        }

        Tensor result = inputs[0];

        for (int i = 1; i < inputs.length; i++) {
            result = result.concatGrad(inputs[i]);
        }

        return result;
    }
}
