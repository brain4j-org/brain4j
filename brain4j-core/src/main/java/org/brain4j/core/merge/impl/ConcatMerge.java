package org.brain4j.core.merge.impl;

import org.brain4j.core.merge.MergeStrategy;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;
import org.brain4j.math.tensor.autograd.operations.ConcatOperation;

public class ConcatMerge implements MergeStrategy {
    @Override
    public Tensor process(Tensor... inputs) {
        if (inputs.length < 2) {
            throw new IllegalArgumentException("ConcatMerge supports at least two input tensors.");
        }

        Tensor result = inputs[0];

        for (int i = 1; i < inputs.length; i++) {
            result = result.concatGrad(inputs[i]);
        }

        return result;
    }

    @Override
    public Tensor inverse(int[] dimensions, Tensor input) {
        return null;
    }
}
