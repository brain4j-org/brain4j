package org.brain4j.math.gpu.memory;

import org.brain4j.math.gpu.TempObject;
import org.silicon.api.device.ComputeBuffer;

public class TempBuffer extends TempObject<ComputeBuffer> {

    public TempBuffer(ComputeBuffer value) {
        super(value, value::free);
    }

    public ComputeBuffer buffer() {
        return value();
    }
}
