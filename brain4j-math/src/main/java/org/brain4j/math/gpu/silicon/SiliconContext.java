package org.brain4j.math.gpu.silicon;

import org.silicon.computing.ComputeQueue;
import org.silicon.kernel.ComputeFunction;
import org.silicon.kernel.ComputeModule;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class SiliconContext {

    private static final Map<SiliconDevice, Map<String, ComputeFunction>> kernelCache = new ConcurrentHashMap<>();
    private static final Map<SiliconDevice, Map<String, ComputeModule>> moduleCache = new ConcurrentHashMap<>();

    private SiliconContext() {}

    public static void register(SiliconDevice device, String kernelName, ComputeFunction function) {
        kernelCache.computeIfAbsent(device, d -> new ConcurrentHashMap<>())
            .put(kernelName, function);
    }

    public static void register(SiliconDevice device, String kernelName, ComputeModule module) {
        try {
            ComputeFunction function = module.getFunction(kernelName);
            register(device, kernelName, function);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to register kernel " + kernelName, e);
        }
    }

    public static void registerAll(SiliconDevice device, ComputeModule module, String... kernelNames) {
        for (String kernelName : kernelNames) {
            register(device, kernelName, module);
        }
    }

    public static void storeModule(SiliconDevice device, String moduleName, ComputeModule module) {
        moduleCache.computeIfAbsent(device, d -> new ConcurrentHashMap<>())
            .put(moduleName, module);
    }

    public static ComputeModule getModule(SiliconDevice device, String moduleName) {
        Map<String, ComputeModule> deviceModules = moduleCache.get(device);
        return deviceModules != null ? deviceModules.get(moduleName) : null;
    }

    public static ComputeFunction findFunction(SiliconDevice device, String kernelName) {
        Map<String, ComputeFunction> deviceKernels = kernelCache.get(device);

        if (deviceKernels == null) {
            throw new IllegalStateException("No kernels registered for device: " + device);
        }

        ComputeFunction function = deviceKernels.get(kernelName);

        if (function == null) {
            throw new IllegalStateException("Kernel " + kernelName + " not registered for device: " + device.getName());
        }

        return function;
    }

    public static QueueHandle getOrCreateQueue(SiliconDevice device) {
        ComputeQueue queue = device.getQueue();

        if (queue != null) {
            return new QueueHandle(queue, false);
        }

        return new QueueHandle(device.newQueue(), true);
    }

    public static void finishAndRelease(ComputeQueue queue) {
        try {
            queue.awaitCompletion();
            queue.release();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to finish and release queue", e);
        }
    }

    public static void clearCache(SiliconDevice device) {
        kernelCache.remove(device);
        moduleCache.remove(device);
    }

    public static void clearAllCaches() {
        kernelCache.clear();
        moduleCache.clear();
    }

    public record QueueHandle(
            ComputeQueue queue,
            boolean temporary
    ) implements AutoCloseable {
        @Override
        public void close() {
            if (temporary) {
                finishAndRelease(queue);
            }
        }
    }
}

