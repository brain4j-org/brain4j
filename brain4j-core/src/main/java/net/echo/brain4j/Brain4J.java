package net.echo.brain4j;

import net.echo.math4j.math.tensor.TensorFactory;
import net.echo.math4j.math.tensor.impl.TensorGPU;

/**
 * Welcome to Brain4J!
 * This class contains various utilities to help you use the library.
 * You can find the documentation related <a href="https://github.com/xEcho1337/brain4j/wiki">here</a>.
 * 
 * @author <a href="https://github.com/xEcho1337">xEcho1337</a>
 * @author <a href="https://github.com/Adversing">Adversing</a>
 */
public class Brain4J {

    private static boolean logging;

    /**
     * Gets the current version of Brain4J.
     * @return The current version.
     */
    public static String version() {
        return "2.7.0";
    }

    /**
     * Checks if the GPU is available.
     * @return True if the GPU is available, false otherwise.
     */
    public static boolean isGPUAvailable() {
        return TensorGPU.isGpuAvailable();
    }

    /**
     * Checks if the GPU is being used.
     * @return True if the GPU is being used, false otherwise.
     */
    public static boolean isUsingGPU() {
        return TensorFactory.isUsingGPU();
    }

    /**
     * Enables the GPU if it's available.
     */
    public static void useGPUIfAvailable() {
        TensorFactory.useGPUIfAvailable();
    }

    /**
     * Forces the CPU to be used.
     */
    public static void useCPU() {
        TensorFactory.forceCPU();
    }

    /**
     * Enables or disables the logging while training.
     * @param logging True to enable logging, false to disable it.
     */
    public static void setLogging(boolean logging) {
        Brain4J.logging = logging;
    }

    /**
     * Checks if the logging is enabled.
     * @return True if the logging is enabled, false otherwise.
     */
    public static boolean isLogging() {
        return logging;
    }
}
