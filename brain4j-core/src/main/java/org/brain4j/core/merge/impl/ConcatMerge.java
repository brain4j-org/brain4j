package org.brain4j.core.merge.impl;

import org.brain4j.core.merge.MergeStrategy;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;
import org.brain4j.math.tensor.autograd.operations.ConcatOperation;

public class ConcatMerge implements MergeStrategy {
    @Override
    public Tensor process(Tensor... inputs) {
        Operation operation = new ConcatOperation();
        Tensor result = operation.forward(inputs);

        result.withGrad();
        result.autogradContext().setOperation(operation, inputs);

        return result;
    }
}
