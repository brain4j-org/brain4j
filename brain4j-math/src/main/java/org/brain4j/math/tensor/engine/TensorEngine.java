package org.brain4j.math.tensor.engine;

import org.brain4j.math.tensor.DType;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;

public interface TensorEngine {

    Tensor create(Shape shape, DType dtype, float... data);

    default Tensor zeros(Shape shape, DType dtype) {
        return create(shape, dtype, new float[shape.size()]);
    }
}
