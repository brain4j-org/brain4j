package org.brain4j.math.gpu.kernel;

import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.gpu.device.Device;
import org.silicon.api.device.ComputeBuffer;
import org.silicon.api.function.ComputeFunction;
import org.silicon.api.kernel.ComputeArgs;
import org.silicon.api.kernel.ComputeQueue;
import org.silicon.api.kernel.ComputeSize;

/**
 * Kernel argument builder and dispatch helper backed by Silicon.
 */
public record KernelFactory(ComputeFunction function, ComputeArgs args) {

    public KernelFactory(ComputeFunction function) {
        this(function, ComputeArgs.of());
    }

    public static KernelFactory create(Device device, String kernelName) {
        return create(GpuContext.findFunction(device, kernelName));
    }

    public static KernelFactory create(ComputeFunction function) {
        return new KernelFactory(function);
    }

    public KernelFactory buffer(ComputeBuffer buffer) {
        args.buffer(buffer);
        return this;
    }

    public KernelFactory intVal(int value) {
        args.intVal(value);
        return this;
    }

    public KernelFactory floatVal(float value) {
        args.floatVal(value);
        return this;
    }

    public KernelFactory longVal(long value) {
        args.longVal(value);
        return this;
    }

    public KernelFactory doubleVal(double value) {
        args.doubleVal(value);
        return this;
    }

    public void launch(ComputeQueue queue, ComputeSize globalSize, ComputeSize localSize) {
        try {
            ComputeSize dispatchGlobalSize = roundGlobalSize(globalSize, localSize);
            queue.dispatch(function, dispatchGlobalSize, localSize, args);
            if (Boolean.getBoolean("brain4j.gpu.sync")) {
                queue.await();
            }
        } catch (Throwable e) {
            throw new RuntimeException(
                "Failed to launch kernel " + function + " with global=" + globalSize + ", local=" + localSize, e
            );
        }
    }

    public void launch(GpuContext.QueueHandle queue, ComputeSize globalSize) {
        launch(queue.queue(), globalSize);
    }

    public void launch(ComputeQueue queue, ComputeSize globalSize) {
        int defaultWorkDim = 256;
        int localX = Math.min(defaultWorkDim, globalSize.x());
        int localY = globalSize.y() > 1 ? Math.min(16, globalSize.y()) : 1;
        int localZ = globalSize.z() > 1 ? Math.min(4, globalSize.z()) : 1;
        launch(queue, globalSize, new ComputeSize(localX, localY, localZ));
    }

    public void launch(ComputeQueue queue, int globalX) {
        launch(queue, new ComputeSize(globalX, 1, 1));
    }

    public void launch(ComputeQueue queue, int globalX, int globalY) {
        launch(queue, new ComputeSize(globalX, globalY, 1));
    }

    public void launch(ComputeQueue queue, int globalX, int globalY, int globalZ) {
        launch(queue, new ComputeSize(globalX, globalY, globalZ));
    }

    public void launch(GpuContext.QueueHandle queueHandle, ComputeSize globalSize, ComputeSize localSize) {
        launch(queueHandle.queue(), globalSize, localSize);
    }

    public void launchAndWait(ComputeQueue queue, ComputeSize globalSize, ComputeSize localSize) {
        try {
            queue.dispatch(function, globalSize, localSize, args);
            queue.await();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to launch kernel and wait", e);
        }
    }

    public ComputeArgs args() {
        return args;
    }

    public ComputeFunction function() {
        return function;
    }

    private static ComputeSize roundGlobalSize(ComputeSize globalSize, ComputeSize localSize) {
        return new ComputeSize(
            roundUp(globalSize.x(), localSize.x()),
            roundUp(globalSize.y(), localSize.y()),
            roundUp(globalSize.z(), localSize.z())
        );
    }

    private static int roundUp(int global, int local) {
        return ((global + local - 1) / local) * local;
    }

    private static ComputeSize toSize(int workDim, long[] values) {
        if (workDim < 1 || workDim > 3) {
            throw new IllegalArgumentException("Work dimension must be 1, 2, or 3");
        }
        if (values.length < workDim) {
            throw new IllegalArgumentException("Expected at least " + workDim + " size values");
        }

        int x = toPositiveInt(values[0], "x");
        int y = workDim > 1 ? toPositiveInt(values[1], "y") : 1;
        int z = workDim > 2 ? toPositiveInt(values[2], "z") : 1;
        return new ComputeSize(x, y, z);
    }

    private static int toPositiveInt(long value, String component) {
        if (value <= 0 || value > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Invalid " + component + " compute size: " + value);
        }
        return (int) value;
    }
}
