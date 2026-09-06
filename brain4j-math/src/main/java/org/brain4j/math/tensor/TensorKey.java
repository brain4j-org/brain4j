package org.brain4j.math.tensor;

import java.util.Arrays;

public record TensorKey(Usage usage, int... shape) {

    public TensorKey {
        shape = shape == null ? new int[0] : shape.clone();
    }

    @Override
    public int[] shape() {
        return shape.clone();
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (!(obj instanceof TensorKey other)) {
            return false;
        }
        return usage == other.usage && Arrays.equals(shape, other.shape);
    }

    @Override
    public int hashCode() {
        int result = usage != null ? usage.hashCode() : 0;
        result = 31 * result + Arrays.hashCode(shape);
        return result;
    }
}
