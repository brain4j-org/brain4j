package org.brain4j.math.gpu.device;

import org.brain4j.math.commons.Commons;
import org.silicon.api.Silicon;
import org.silicon.api.backend.BackendType;
import org.silicon.api.backend.ComputeBackend;
import org.silicon.api.device.ComputeDevice;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;

import static java.nio.charset.StandardCharsets.UTF_8;

/**
 * Device discovery helpers backed by the Silicon runtime.
 */
public class DeviceUtils {

    private DeviceUtils() {}

    public static Device findDevice() {
        return findDevice(0);
    }

    public static Device findDevice(int index) {
        try {
            return new Device(index);
        } catch (Exception e) {
            return null;
        }
    }

    public static Device findDevice(String name) {
        if (name == null) {
            return findDevice(0);
        }

        for (int i = 0; i < 8; i++) {
            Device device = null;
            try {
                device = new Device(i);

                if (device.name().toLowerCase().contains(name.toLowerCase())) {
                    return device;
                }

                freeQuietly(device);
            } catch (Exception e) {
                freeQuietly(device);
                break;
            }
        }

        return null;
    }

    public static List<String> allDeviceNames() {
        ComputeBackend backend = Silicon.backend();

        if (backend == null) {
            throw new IllegalStateException("Backend is null! Make sure to import at least one backend!");
        }

        List<String> names = new ArrayList<>();

        int count = backend.deviceCount();
        for (int i = 0; i < count; i++) {
            try {
                ComputeDevice device = Silicon.createDevice(i);
                names.add(device.name());
            } catch (Throwable e) {
                break;
            }
        }

        return names;
    }

    public static BackendType getBackendType() {
        return Silicon.backend().type();
    }

    public static ComputeBackend getBackend() {
        return Silicon.backend();
    }

    public static void chooseBackend(BackendType backendType) {
        Silicon.chooseBackend(backendType);
    }

    public static String readKernelSource(String resourcePath) {
        try (InputStream input = DeviceUtils.class.getResourceAsStream(resourcePath)) {
            if (input == null) {
                throw Commons.illegalArgument("Resource not found: %s", resourcePath);
            }
            return new String(input.readAllBytes(), UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to read kernel source from: " + resourcePath, e);
        }
    }

    public static boolean isSimdAvailable() {
        return ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent();
    }

    public static String getErrorCode(int code) {
        return code == 0 ? "SUCCESS" : "Silicon backend error code " + code;
    }

    public static void checkError(String profiler, int err) {
        if (err == 0) return;
        throw new RuntimeException("GPU(" + profiler + ") - " + getErrorCode(err));
    }

    private static UnsupportedOperationException unsupportedRawGpuHandle() {
        return new UnsupportedOperationException(
            "Raw native device handles are not available in the Silicon GPU backend"
        );
    }

    private static void freeQuietly(Device device) {
        if (device == null) {
            return;
        }

        try {
            device.free();
        } catch (RuntimeException ignored) {
        }
    }
}
