package org.brain4j.math.tensor.autograd.operations;

import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.autograd.Operation;
import org.brain4j.math.tensor.index.Range;

public class ConcatOperation implements Operation {

    @Override
    public Tensor forward(Tensor... inputs) {
        return inputs[0].concat(inputs[1]);
    }

    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor... inputs) {
        if (inputs.length != 2) {
            throw new IllegalArgumentException("ConcatOperation supports exactly two input tensors.");
        }

        Tensor a = inputs[0];
        Tensor b = inputs[1];

        int[] shapeA = a.shape();
        int[] shapeB = b.shape();
        int rank = shapeA.length;
        int lastDim = rank - 1;

        int sizeA = shapeA[lastDim];
        int sizeB = shapeB[lastDim];

        Range[] base = new Range[rank];
        for (int i = 0; i < lastDim; i++) {
            base[i] = Range.all();
        }

        Range[] rangeA = base.clone();
        Range[] rangeB = base.clone();

        rangeA[lastDim] = new Range(0, sizeA);
        rangeB[lastDim] = new Range(sizeA, sizeA + sizeB);

        Tensor gradA = gradOutput.slice(rangeA);
        Tensor gradB = gradOutput.slice(rangeB);

        return new Tensor[]{gradA, gradB};
    }
}
