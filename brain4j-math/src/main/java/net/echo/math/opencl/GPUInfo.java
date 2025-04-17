package net.echo.math.opencl;

public record GPUInfo(String name, String vendor, String version, long globalMemory, long localMemory, int maxWorkGroupSize) {
}
