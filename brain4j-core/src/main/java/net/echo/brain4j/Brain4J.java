package net.echo.brain4j;

import net.echo.math4j.math.tensor.TensorFactory;
import net.echo.math4j.math.tensor.impl.TensorGPU;

public class Brain4J {

    public static String version() {
        return "2.7.0";
    }

    public static boolean isGPUAvailable() {
        return TensorGPU.isGpuAvailable();
    }

    public static boolean isUsingGPU() {
        return TensorFactory.isUsingGPU();
    }

    public static void useGPUIfAvailable() {
        TensorFactory.useGPUIfAvailable();
    }

    public static void useCPU() {
        TensorFactory.forceCPU();
    }
}
