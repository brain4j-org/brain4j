package org.brain4j.math.tensor;

import java.util.Arrays;

public record TensorLayout(int[] shape, int[] strides, int offset) {

    public TensorLayout {
        if (shape == null || strides == null || shape.length != strides.length) {
            throw new IllegalArgumentException("Shape and strides must have the same rank");
        }
        shape = shape.clone();
        strides = strides.clone();
        for (int dimension : shape) {
            if (dimension < 0) {
                throw new IllegalArgumentException("Shape dimensions cannot be negative");
            }
        }
    }

    @Override
    public int[] shape() {
        return shape.clone();
    }

    @Override
    public int[] strides() {
        return strides.clone();
    }

    public int rank() {
        return shape.length;
    }

    public int elements() {
        int result = 1;
        for (int dimension : shape) {
            result = Math.multiplyExact(result, dimension);
        }
        return result;
    }

    public int linearIndex(int... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException("Expected " + shape.length + " indices, got " + indices.length);
        }
        int result = offset;
        for (int i = 0; i < indices.length; i++) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException("Index " + indices[i] + " outside dimension " + i);
            }
            result += indices[i] * strides[i];
        }
        return result;
    }

    public boolean isContiguous() {
        int expected = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (shape[i] > 1 && strides[i] != expected) {
                return false;
            }
            expected *= shape[i];
        }
        return true;
    }

    @Override
    public String toString() {
        return "TensorLayout{shape=" + Arrays.toString(shape) + ", strides="
                + Arrays.toString(strides) + ", offset=" + offset + "}";
    }
}
