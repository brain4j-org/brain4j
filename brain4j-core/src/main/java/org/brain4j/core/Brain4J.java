package org.brain4j.core;

import ch.qos.logback.classic.Level;
import org.brain4j.core.activation.Activations;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.device.Device;
import org.brain4j.math.device.DeviceType;
import org.brain4j.math.device.DeviceUtils;
import org.brain4j.math.kernel.GpuContextHandler;
import org.brain4j.math.tensor.impl.cpu.CpuTensor;
import org.jocl.cl_context;
import org.jocl.cl_program;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Entry point for the Brain4J machine learning framework.
 * <p>
 * The {@code Brain4J} class provides central static methods to access core functionalities
 * such as framework initialization, device discovery, and version management.
 * <p>
 * This class is designed to serve as the main API access point, and will be extended
 * in the future to include global configuration, logging, and other utilities.
 *
 * <h2>Usage</h2>
 * Before using any tensor operations or training functionality, call {@link #initialize()}
 * to ensure proper device setup and runtime configuration.
 *
 * <pre>{@code
 *     Brain4J.initialize();
 * }</pre>
 *
 * @since 3.0
 * @author xEcho1337
 * @author Adversing
 */
public class Brain4J {

    private static final Logger logger = LoggerFactory.getLogger(Brain4J.class);
    private static boolean logging;

    public static String version() {
        return "3.0";
    }

    public static boolean logging() {
        return logging;
    }

    public static void disableLogging() {
        ch.qos.logback.classic.Logger root = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
        root.setLevel(Level.OFF);
        logging = false;
    }

    public static void enableLogging() {
        ch.qos.logback.classic.Logger root = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
        root.setLevel(Level.DEBUG);
        logging = true;
    }

    public static void initialize() {
        CpuTensor.initialize();

        logging = true;

        logger.info("Brain4J v{} initialized.", version());
    }

    public static String availableDevices() {
        return String.join(", ", DeviceUtils.allDeviceNames());
    }

    public static void useDevice(DeviceType deviceType, String deviceName) {
        Device device = DeviceUtils.findDevice(deviceType, deviceName);

        if (device == null) {
            throw new IllegalArgumentException("No such device: " + deviceName);
        }

        useDevice(device);
    }

    public static void useDevice(Device device) {
        DeviceUtils.setCurrentDevice(device);

        cl_context context = device.context();
        cl_program activationsProgram = DeviceUtils.createBuildProgram(context, "/kernels/basic/activations.cl");
        cl_program gradientClipProgram = DeviceUtils.createBuildProgram(context, "/kernels/basic/gradient_clippers.cl");

        for (Activations activation : Activations.values()) {
            Activation function = activation.function();
            String prefix = function.kernelPrefix();

            GpuContextHandler.register(prefix + "_forward", activationsProgram);
            GpuContextHandler.register(prefix + "_backward", activationsProgram);
        }

        GpuContextHandler.register("hard_clip", gradientClipProgram);
        GpuContextHandler.register("l2_clip", gradientClipProgram);
    }

    public static Device currentDevice() {
        return DeviceUtils.currentDevice();
    }

    public static Device findDevice(String deviceName) {
        return DeviceUtils.findDevice(DeviceType.GPU, deviceName);
    }
}
