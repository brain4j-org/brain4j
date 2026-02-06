package org.brain4j.math.operators.impl;

import org.brain4j.math.operators.OperatorSpec;
import org.brain4j.math.operators.ShapeException;
import org.brain4j.math.tensor.Shape;

public class MatMulOperator implements OperatorSpec {
    @Override
    public int inputCount() {
        return 2;
    }

    @Override
    public int outputCount() {
        return 1;
    }

    @Override
    public void validateInputShapes(Shape... inputShapes) throws ShapeException {
        Shape a = inputShapes[0]; // [..., M, N]
        Shape b = inputShapes[1]; // [..., N, P]

        if (a.rank() < 2) throw new ShapeException("A must at least be rank 2");
        if (b.rank() < 2) throw new ShapeException("B must at least be rank 2");

        int NA = a.last();
        int NB = b.last(1);

        if (NA != NB) throw new ShapeException("Inner dimensions do not match: %s != %s".formatted(NA, NB));

        int i = a.rank() - 3;
        int j = b.rank() - 3;

        while (i >= 0 || j >= 0) {
            int da = (i >= 0) ? a.dim(i) : 1;
            int db = (j >= 0) ? b.dim(j) : 1;

            if (da != db && da != 1 && db != 1) {
                throw new ShapeException("Batch dims not broadcastable");
            }

            i--;
            j--;
        }
    }

    @Override
    public Shape inferOutputShape(int outputIndex, Shape... inputShapes) {
        Shape a = inputShapes[0]; // [..., M, K]
        Shape b = inputShapes[1]; // [..., K, P]

        int rA = a.rank();
        int rB = b.rank();

        int M = a.last(1);
        int P = b.last();

        int outRank = Math.max(rA, rB);
        int leadingCount = Math.max(0, outRank - 2);

        int[] outShape = new int[outRank];

        for (int offset = 0; offset < leadingCount; offset++) {
            int ia = (rA - 3) - offset;
            int ib = (rB - 3) - offset;

            int da = (ia >= 0) ? a.dim(ia) : 1;
            int db = (ib >= 0) ? b.dim(ib) : 1;

            outShape[leadingCount - 1 - offset] = Math.max(da, db);
        }

        outShape[outRank - 2] = M; // ... M
        outShape[outRank - 1] = P; // ... P

        return Shape.of(outShape);
    }


    @Override
    public boolean isInPlace(int inputIndex, int outputIndex) {
        return false;
    }
}
