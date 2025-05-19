package org.brain4j.core;

import org.brain4j.math.device.DeviceType;
import org.brain4j.math.exceptions.NativeException;
import org.brain4j.math.tensor.Tensors;
import org.brain4j.math.tensor.impl.TensorCPU;
import org.brain4j.math.tensor.impl.TensorGPU;

import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

public class Brain4J {

    public static String version() {
        return "3.0";
    }

    public static void initialize(DeviceType deviceType) {
        TensorCPU.initializeCPU();

        try {
            TensorGPU.initializeGPU();
        } catch (NativeException e) {
            e.printStackTrace(System.err);
        }

        useDevice(deviceType);
        System.setOut(new PrintStream(System.out, true, StandardCharsets.UTF_8));
    }

    public static void useDevice(DeviceType deviceType) {
        switch (deviceType) {
            case CPU -> Tensors.forceCPU();
            case GPU -> Tensors.useGPUIfAvailable();
            default -> throw new IllegalArgumentException("Unsupported device type: " + deviceType);
        }
    }
}
