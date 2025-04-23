package net.echo.math.device;

public enum DeviceType {

    DEFAULT(1),
    CPU(1 << 1),
    GPU(1 << 2),
    ACCELERATOR(1 << 3),
    CUSTOM(1 << 4);

    private final long mask;

    DeviceType(long mask) {
        this.mask = mask;
    }

    public long getMask() {
        return mask;
    }
}
