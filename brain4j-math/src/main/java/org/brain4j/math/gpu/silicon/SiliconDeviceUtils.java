package org.brain4j.math.gpu.silicon;

import org.silicon.api.Silicon;
import org.silicon.api.backend.BackendType;
import org.silicon.api.backend.ComputeBackend;
import org.silicon.api.device.ComputeDevice;

import java.util.ArrayList;
import java.util.List;

public class SiliconDeviceUtils {

    private SiliconDeviceUtils() {}

    public static SiliconDevice findDevice() {
        return findDevice(0);
    }

    public static SiliconDevice findDevice(int index) {
        try {
            return new SiliconDevice(index);
        } catch (Exception e) {
            return null;
        }
    }

    public static SiliconDevice findDevice(String name) {
        if (name == null) {
            return findDevice(0);
        }

        // 0-7 array indexing to find an available device
        for (int i = 0; i < 8; i++) {
            try {
                SiliconDevice device = new SiliconDevice(i);
                
                if (device.getName().toLowerCase().contains(name.toLowerCase())) {
                    return device;
                }
            } catch (Exception e) {
                break; // there are no more devices to check
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
        
        for (int i = 0; i < count; i++) { // the same here
            try {
                ComputeDevice device = Silicon.createDevice(i);
                names.add(device.name());
            } catch (Throwable e) {
                break; // there are no more devices
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

    public static boolean isSimdAvailable() {
        return ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent();
    }
}

