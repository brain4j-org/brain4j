package org.brain4j.math.gpu;

import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.memory.GpuQueue;
import org.silicon.api.function.ComputeFunction;
import org.silicon.api.function.ComputeModule;
import org.silicon.api.kernel.ComputeQueue;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Kernel and module registry for Brain4J GPU execution.
 */
public class GpuContext {

    private static final Map<Device, Map<String, ComputeFunction>> KERNEL_CACHE = new ConcurrentHashMap<>();
    private static final Map<Device, Map<String, ComputeModule>> MODULE_CACHE = new ConcurrentHashMap<>();

    private GpuContext() {}

    public static void register(Device device, String kernelName, ComputeFunction function) {
        KERNEL_CACHE.computeIfAbsent(device, d -> new ConcurrentHashMap<>())
            .put(kernelName, function);
    }

    public static void register(Device device, String kernelName, ComputeModule module) {
        try {
            ComputeFunction function = module.getFunction(kernelName);
            register(device, kernelName, function);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to register kernel " + kernelName, e);
        }
    }

    public static void registerAll(Device device, ComputeModule module, String... kernelNames) {
        for (String kernelName : kernelNames) {
            register(device, kernelName, module);
        }
    }

    public static void storeModule(Device device, String moduleName, ComputeModule module) {
        MODULE_CACHE.computeIfAbsent(device, d -> new ConcurrentHashMap<>())
            .put(moduleName, module);
    }

    public static ComputeModule getModule(Device device, String moduleName) {
        Map<String, ComputeModule> deviceModules = MODULE_CACHE.get(device);
        return deviceModules != null ? deviceModules.get(moduleName) : null;
    }

    public static ComputeFunction findFunction(Device device, String kernelName) {
        Map<String, ComputeFunction> deviceKernels = KERNEL_CACHE.get(device);

        if (deviceKernels == null) {
            throw new IllegalStateException("No kernels registered for device: " + device);
        }

        ComputeFunction function = deviceKernels.get(kernelName);

        if (function == null) {
            throw new IllegalStateException("Kernel " + kernelName + " not registered for device: " + device.getName());
        }

        return function;
    }

    @Deprecated
    public static long findKernel(Device device, String kernelName) {
        throw new UnsupportedOperationException(
            "Raw native kernels are not available in the Silicon GPU backend"
        );
    }

    public static QueueHandle getOrCreateQueue(Device device) {
        ComputeQueue queue = device.queue();
        if (queue != null) {
            return new QueueHandle(queue, false);
        }

        return new QueueHandle(device.context().createQueue(), true);
    }

    public static GpuQueue getOrCreate(Device device) {
        GpuQueue queue = device.getQueue();
        if (queue != null) {
            return queue;
        }

        return new GpuQueue(device.context().createQueue(), true);
    }

    public static void finishAndRelease(GpuQueue queue) {
        if (queue == null) return;
        queue.close();
    }

    @Deprecated
    public static void finishAndRelease(long commandQueue) {
        throw new UnsupportedOperationException(
            "Raw native command queues are not available in the Silicon GPU backend"
        );
    }

    public static void finishAndRelease(Device device) {
        GpuQueue queue = device.getQueue();
        if (queue != null) {
            queue.close();
        }
        device.setQueue((GpuQueue) null);
    }

    @Deprecated
    public static void register(Device device, String kernelName, long program) {
        throw new UnsupportedOperationException(
            "Native program registration has been removed. Register Silicon ComputeFunction instances instead."
        );
    }

    public static void clearCache(Device device) {
        KERNEL_CACHE.remove(device);
        MODULE_CACHE.remove(device);
    }

    public static void clearAllCaches() {
        KERNEL_CACHE.clear();
        MODULE_CACHE.clear();
    }

    public record QueueHandle(ComputeQueue queue, boolean temporary) implements AutoCloseable {
        @Override
        public void close() {
            if (temporary) {
                queue.await();
                queue.free();
            }
        }
    }
}
