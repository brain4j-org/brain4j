package net.echo.brain4j;

import net.echo.math.BrainUtils;
import net.echo.math.opencl.GPUInfo;
import net.echo.math.tensor.Tensors;
import net.echo.math.tensor.impl.TensorGPU;

import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

/**
 * Welcome to Brain4J!
 * This class contains various utilities to help you use the library.
 * You can find the documentation related <a href="https://github.com/xEcho1337/brain4j/wiki">here</a>.
 * 
 * @author <a href="https://github.com/xEcho1337">xEcho1337</a>
 * @author <a href="https://github.com/Adversing">Adversing</a>
 */
public class Brain4J {

    private static boolean initialized;
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
        return Tensors.isUsingGPU();
    }

    /**
     * Enables the GPU if it's available.
     */
    public static void useGPUIfAvailable() {
        Tensors.useGPUIfAvailable();
    }

    /**
     * Forces the CPU to be used.
     */
    public static void useCPU() {
        Tensors.forceCPU();
    }

    /**
     * Enables or disables the logging while training.
     * @param logging True to enable logging, false to disable it.
     */
    public static void setLogging(boolean logging) {
        Brain4J.logging = logging;

        if (logging) {
            // Required to make the progress bar work on windows
            System.setOut(new PrintStream(System.out, true, StandardCharsets.UTF_8));
        }
    }

    /**
     * Gets the header char for the progress bar. Initializes the print stream if needed.
     * @return The header char.
     */
    public static String getHeaderChar() {
        if (!initialized) {
            System.setOut(new PrintStream(System.out, true, StandardCharsets.UTF_8));
            initialized = true;
        }

        return "━";
    }

    /**
     * Prints the device info of the GPU.
     * @param info The GPU info.
     */
    public static void printDeviceInfo(GPUInfo info) {
        String builder = BrainUtils.getHeader(" Device Information ", getHeaderChar()) +
                "Name: " + info.name() + "\n" +
                "Vendor: " + info.vendor() + "\n" +
                "OpenCL Version: " + info.version() + "\n" +
                "Global Memory: " + info.globalMemory() / (1024 * 1024) + " MB\n" +
                "Local Memory: " + info.localMemory() / 1024 + " KB\n" +
                "Max Work Group Size: " + info.maxWorkGroupSize() + "\n";

        System.out.println(builder);
    }

    /**
     * Checks if the logging is enabled.
     * @return True if the logging is enabled, false otherwise.
     */
    public static boolean isLogging() {
        return logging;
    }
}
