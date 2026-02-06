package org.brain4j.math.tensor;

import org.brain4j.math.commons.Commons;

public class Shape {

    private final int[] dims;

    private Shape(int[] dims) {
        for (int dim : dims) {
            if (dim < 0) {
                throw Commons.illegalArgument("Dimension at %s is negative!", dim);
            }
        }

        this.dims = dims.clone();
    }

    public static Shape of(int... dims) {
        return new Shape(dims);
    }

    public int dim(int index) {
        return dims[index];
    }

    public int last() {
        return dims[dims.length - 1];
    }

    public int last(int offset) {
        if (offset > dims.length - 1) {
            throw new IllegalArgumentException("Offset cannot be higher than rank - 1");
        }

        return dims[dims.length - 1 - offset];
    }

    public int rank() {
        return dims.length;
    }

    public int[] dims() {
        return dims;
    }

    public Shape slice(int start, int end) {
        if (end <= start) {
            throw new IllegalArgumentException("End must be greater than start!");
        }
        int[] result = new int[end - start];
        System.arraycopy(dims, start, result, 0, result.length);
        return Shape.of(result);
    }
}
