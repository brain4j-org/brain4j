package org.brain4j.math.tensor.storage;

import org.brain4j.math.Copyable;
import org.brain4j.math.tensor.DType;
import org.brain4j.math.tensor.TensorLayout;

/** Shared ownership of tensor memory. Views reference the same storage. */
public interface TensorStorage extends Copyable<TensorStorage> {

    DType dtype();

    int elements();

    float getFloat(int index);

    void setFloat(int index, float value);

    TensorStorage copy(TensorLayout layout);
}
