package org.brain4j.math.tensor.storage.impl;

import java.util.Arrays;

import org.brain4j.math.tensor.DType;
import org.brain4j.math.tensor.TensorLayout;
import org.brain4j.math.tensor.storage.TensorStorage;

public record CpuStorage(DType dtype, float[] data) implements TensorStorage {

    public CpuStorage(int elements) {
        this(DType.FLOAT32, new float[elements]);
    }

    public CpuStorage(DType dtype, int elements) {
        this(dtype, new float[elements]);
    }

    public CpuStorage {
        if (dtype == null || data == null) {
            throw new NullPointerException("dtype and data are required");
        }
    }

    @Override
    public int elements() {
        return data.length;
    }

    @Override
    public float getFloat(int index) {
        return data[index];
    }

    @Override
    public void setFloat(int index, float value) {
        data[index] = value;
    }

    @Override
    public CpuStorage copy() {
        return new CpuStorage(dtype, Arrays.copyOf(data, data.length));
    }

    @Override
    public CpuStorage copy(TensorLayout layout) {
        float[] result = new float[layout.elements()];
        int[] shape = layout.shape();
        int[] strides = layout.strides();
        int[] indices = new int[layout.rank()];
        int physicalIndex = layout.offset();

        for (int linear = 0; linear < result.length; linear++) {
            result[linear] = data[physicalIndex];
            if (linear + 1 < result.length) {
                physicalIndex = advanceIndex(indices, shape, strides, physicalIndex);
            }
        }

        return new CpuStorage(dtype, result);
    }

    public static int advanceIndex(int[] indices, int[] shape, int[] strides, int physicalIndex) {
        for (int dimension = shape.length - 1; dimension >= 0; dimension--) {
            indices[dimension]++;
            physicalIndex += strides[dimension];

            if (indices[dimension] < shape[dimension]) {
                return physicalIndex;
            }

            indices[dimension] = 0;
            physicalIndex -= shape[dimension] * strides[dimension];
        }

        return physicalIndex;
    }
}
