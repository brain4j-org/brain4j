package org.brain4j.core;

import org.brain4j.datasets.api.HuggingFaceClient;
import org.brain4j.math.device.DeviceUtils;
import org.brain4j.math.tensor.impl.TensorCPU;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

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

    public static String version() {
        return "3.0";
    }

    public static void initialize() {
        TensorCPU.initialize();
        logger.info("Brain4J v{} initialized.", version());
        logger.info("Available devices: {}", availableDevices());
        System.setOut(new PrintStream(System.out, true, StandardCharsets.UTF_8));
    }

    public static String availableDevices() {
        return String.join(", ", DeviceUtils.getAllDeviceNames());
    }
}
