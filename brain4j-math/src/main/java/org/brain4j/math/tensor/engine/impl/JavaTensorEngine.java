package org.brain4j.math.tensor.engine.impl;

import java.util.Objects;

import org.brain4j.math.tensor.DType;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.engine.TensorEngine;
import org.brain4j.math.tensor.impl.CpuTensor;

// TODO
public final class JavaTensorEngine implements TensorEngine {
    @Override
    public Tensor create(Shape shape, DType dtype, float... data) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(dtype, "dtype");
        Objects.requireNonNull(data, "data");
        if (dtype != DType.FLOAT32) {
            throw new UnsupportedOperationException("CPU engine currently supports FLOAT32 only");
        }
        if (data.length != shape.size()) {
            throw new IllegalArgumentException("Data length does not match shape");
        }
        return new CpuTensor(shape, data);
    }
}
