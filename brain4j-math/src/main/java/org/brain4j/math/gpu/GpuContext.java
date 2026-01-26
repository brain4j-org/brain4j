package org.brain4j.math.gpu;

import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.device.DeviceUtils;
import org.brain4j.math.gpu.memory.GpuQueue;
import org.lwjgl.opencl.CL10;

import java.util.HashMap;
import java.util.Map;

public class GpuContext {

    private static final Map<Device, Map<String, Long>> kernelCache = new HashMap<>();

    public static void register(Device device, String kernelName, long program) {
        kernelCache.computeIfAbsent(device, d -> new HashMap<>())
            .compute(kernelName, (name, existingKernel) -> {
                if (existingKernel != null) return existingKernel;

                int[] err = new int[1];
                long result = CL10.clCreateKernel(program, name, err);
                
                DeviceUtils.checkError("create_kernel", err[0]);
                return result;
            });
    }

    public static long findKernel(Device device, String kernelName) {
        Map<String, Long> deviceKernels = kernelCache.get(device);

        if (deviceKernels == null) {
            throw new IllegalStateException("No kernels registered for device: " + device);
        }

        long kernel = deviceKernels.getOrDefault(kernelName, -1L);

        if (kernel == -1) {
            throw new IllegalStateException("Kernel " + kernelName + " not registered for device: " + device.name());
        }

        return kernel;
    }

    public static GpuQueue getOrCreate(Device device) {
        GpuQueue queue = device.getQueue();
        
        if (queue == null) {
            long clQueue = device.newCommandQueue();
            queue = new GpuQueue(clQueue, true);
        }

        return queue;
    }

    public static void finishAndRelease(long commandQueue) {
        DeviceUtils.checkError("finish", CL10.clFinish(commandQueue));
        DeviceUtils.checkError("release_command_queue", CL10.clReleaseCommandQueue(commandQueue));
    }

    public static void finishAndRelease(Device device) {
        GpuQueue queue = device.getQueue();
        
        if (queue != null && queue.pointer() != 0) {
            finishAndRelease(queue.pointer());
        }
        
        device.setQueue(null);
    }
}