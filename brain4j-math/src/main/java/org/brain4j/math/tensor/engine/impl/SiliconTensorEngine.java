package org.brain4j.math.tensor.engine.impl;

import java.util.Objects;

import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.tensor.DType;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.engine.TensorEngine;

// TODO
public final class SiliconTensorEngine implements TensorEngine {
    private final Device device;

    public SiliconTensorEngine(Device device) {
        this.device = Objects.requireNonNull(device, "device");
    }

    public Device device() {
        return device;
    }

    @Override
    public Tensor create(Shape shape, DType dtype, float... data) {
        throw new UnsupportedOperationException(
                "Silicon tensor allocation is not connected to the tensor storage implementation yet");
    }
}
