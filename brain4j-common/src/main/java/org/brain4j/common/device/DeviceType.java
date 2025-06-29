package org.brain4j.common.device;

public enum DeviceType {

    CPU(1 << 1),
    GPU(1 << 2);

    private final long mask;

    DeviceType(long mask) {
        this.mask = mask;
    }

    public long getMask() {
        return mask;
    }
}
