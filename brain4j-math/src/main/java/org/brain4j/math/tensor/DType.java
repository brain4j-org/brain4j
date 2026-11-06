package org.brain4j.math.tensor;

public enum DType {
    BOOL(1, false),
    INT8(1, true),
    UINT8(1, true),
    INT32(4, true),
    FLOAT16(2, false),
    BFLOAT16(2, false),
    FLOAT32(4, false),
    FLOAT64(8, false);

    private final int bytes;
    private final boolean integral;

    DType(int bytes, boolean integral) {
        this.bytes = bytes;
        this.integral = integral;
    }

    public int bytes() {
        return bytes;
    }

    public boolean isIntegral() {
        return integral;
    }
}
