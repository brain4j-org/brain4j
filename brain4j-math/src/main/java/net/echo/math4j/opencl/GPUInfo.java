package net.echo.math4j.opencl;

public record GPUInfo(String name, String vendor, String version, long globalMemory, long localMemory, int maxWorkGroupSize) {
}
