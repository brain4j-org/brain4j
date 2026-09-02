package org.brain4j.core;

import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.device.DeviceUtils;
import org.brain4j.math.tensor.impl.GpuTensor;

import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Entry point for the Brain4J machine learning framework.
 * <p>
 * The {@code Brain4J} class provides central static methods to access core functionalities
 * such as framework initialization, device discovery, and version management.
 * <p>
 * This class is designed to serve as the main API access point, and will be extended
 * in the future to include global configuration, logging, and other utilities.
 *
 * <p><b>Example usage:</b>
 * <pre>{@code
 * System.out.println("Brain4J version: " + Brain4J.version());
 * Brain4J.enableLogging();
 * System.out.println("Available devices: " + Brain4J.availableDevices());
 * }</pre>
 *
 * @author xEcho1337
 * @author Adversing
 */
public class Brain4J {

    private static boolean logging = true;
    private static boolean disableColors = false;
    private static boolean fixedOutStream = false;
    private static int decimalDigits = 4;
    
    /**
     * Returns the current version of the Brain4J framework.
     *
     * @return the version string (e.g. "3.0")
     */
    public static String getVersion() {
        return "3.0";
    }

    public static void fixConsole() {
        if (fixedOutStream) return;
        
        System.setOut(new PrintStream(System.out, true, StandardCharsets.UTF_8));
        fixedOutStream = true;
    }
    
    /**
     * Indicates whether training progress and system information
     * are currently being logged to the console.
     *
     * @return {@code true} if logging is enabled; {@code false} by default
     */
    public static boolean isLogging() {
        fixConsole();
        return logging;
    }

    /**
     * Sets whether logging should be enabled or disabled during training sessions.
     * <p>
     * When enabled, Brain4J will print progress information such as epoch, current batch and performance evaluations.
     * @param logging {@code true} to enable logging, {@code false} to disable it
     */
    public static void setLogging(boolean logging) {
        Brain4J.logging = logging;
    }
    
    /**
     * Indicates whether the ANSI color codes are active or not.
     * @return {@code true} if the colors are disabled; {@code false} by default
     */
    public static boolean isDisableColors() {
        return disableColors;
    }
    
    /**
     * Sets whether the ANSI color codes should be active.
     * @param disableColors the flag to enable/disable colors
     */
    public static void setDisableColors(boolean disableColors) {
        Brain4J.disableColors = disableColors;
    }
    
    /**
     * Returns the numeric precision used when displaying
     * loss or metric values during training.
     *
     * @return the number of decimal digits displayed (default: 4)
     */
    public static int getDecimalDigits() {
        return decimalDigits;
    }

    /**
     * Sets the numeric precision used when printing loss values.
     * <p>
     * This affects how many decimal digits are shown in logs
     * and formatted console outputs.
     *
     * @param decimalDigits the number of digits to display after the decimal point
     */
    public static void setDecimalDigits(int decimalDigits) {
        Brain4J.decimalDigits = decimalDigits;
    }

    /**
     * Returns a comma-separated list of all available GPU devices
     * detected by the Silicon backend.
     * <p>
     * If no devices are available, an empty string is returned.
     *
     * @return a comma-separated list of device names
     */
    public static String getAvailableDevices() {
        return String.join(", ", DeviceUtils.allDeviceNames());
    }

    /**
     * Initializes GPU kernels on the specified device.
     * <p>
     * This method compiles and loads all GPU-side kernels used
     * by {@link GpuTensor} operations. It should be called before
     * performing any GPU computation if not done automatically.
     *
     * @param device the target GPU device to initialize
     */
    public static void initKernels(Device device) {
        GpuTensor.initKernels(device);
    }
    
    /**
     * Returns the first available GPU device detected on the system.
     * <p>
     * This method is useful when the system contains a single GPU
     * or when the default device is sufficient for computation.
     *
     * @return the first detected {@link Device}
     * @throws IllegalStateException if no GPU devices are found
     */
    public static Device firstDevice() {
        try {
            List<String> devices = DeviceUtils.allDeviceNames();

            if (devices.isEmpty()) {
                return null;
            }

            Device device = DeviceUtils.findDevice(devices.getFirst());

            if (device != null) Brain4J.initKernels(device);

            return device;
        } catch (IllegalStateException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Returns a list of all GPU devices available to the framework.
     * <p>
     * Each {@link Device} object represents a physical or logical
     * compute device accessible through Silicon.
     *
     * @return a list of all available {@link Device} instances
     */
    public static List<Device> getAllDevices() {
        List<Device> devices = new ArrayList<>();

        for (String device : DeviceUtils.allDeviceNames()) {
            devices.add(DeviceUtils.findDevice(device));
        }

        return devices;
    }

    /**
     * Finds a specific GPU device by its name.
     * <p>
     * The search is case-sensitive and matches the full name
     * returned by {@link DeviceUtils#allDeviceNames()}.
     *
     * @param deviceName the name of the device to look for
     * @return the corresponding {@link Device} instance, or {@code null} if not found
     */
    public static Device findDevice(String deviceName) {
        return DeviceUtils.findDevice(deviceName);
    }
}
