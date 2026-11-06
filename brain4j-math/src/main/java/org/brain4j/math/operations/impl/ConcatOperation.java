package org.brain4j.math.operations.impl;

import org.brain4j.math.commons.Range;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.operations.Operation;

import java.util.Arrays;

/**
 * Concatenates two tensors along one dimension. Use it to record joins in the
 * computation graph.
 *
 * @param dimension concatenation axis, -1 selects the last axis
 */
public record ConcatOperation(int dimension) implements Operation {

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor compute(Tensor... inputs) {
        if (dimension == -1) {
            return inputs[0].concat(inputs[1]);
        }

        return inputs[0].concat(inputs[1], dimension);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor[] backward(Tensor gradOutput, Tensor output, Tensor... inputs) {
        if (inputs.length != 2) {
            throw new IllegalArgumentException("ConcatOperation supports exactly two input tensors.");
        }

        Tensor a = inputs[0];
        Tensor b = inputs[1];

        int[] shapeA = a.shape();
        int[] shapeB = b.shape();
        int rank = shapeA.length;
        int actualDim = dimension;

        if (actualDim == -1) {
            actualDim = shapeA.length - 1;
        }

        if (actualDim < 0 || actualDim >= rank) {
            throw new IllegalArgumentException("Invalid concat dimension: " + actualDim);
        }

        int sizeA = shapeA[actualDim];
        int sizeB = shapeB[actualDim];

        Range[] base = new Range[rank];
        Arrays.fill(base, Range.all());

        Range[] rangeA = base.clone();
        rangeA[actualDim] = new Range(0, sizeA);

        Range[] rangeB = base.clone();
        rangeB[actualDim] = new Range(sizeA, sizeA + sizeB);

        Tensor gradA = gradOutput.slice(rangeA);
        Tensor gradB = gradOutput.slice(rangeB);

        return new Tensor[] { gradA, gradB };
    }
}
