package org.brain4j.math.gpu.silicon;

import org.silicon.computing.ComputeArgs;
import org.silicon.computing.ComputeQueue;
import org.silicon.computing.ComputeSize;
import org.silicon.device.ComputeBuffer;
import org.silicon.kernel.ComputeFunction;

public class SiliconKernel {
    
    private final ComputeFunction function;
    private final ComputeArgs args;

    private SiliconKernel(ComputeFunction function) {
        this.function = function;
        this.args = ComputeArgs.of();
    }

    public static SiliconKernel create(SiliconDevice device, String kernelName) {
        ComputeFunction function = SiliconContext.findFunction(device, kernelName);
        return new SiliconKernel(function);
    }

    public static SiliconKernel create(SiliconDevice device, ComputeFunction function) {
        return new SiliconKernel(function);
    }

    public SiliconKernel buffer(SiliconBuffer buffer) {
        args.buffer(buffer.getBuffer());
        return this;
    }

    public SiliconKernel buffer(ComputeBuffer buffer) {
        args.buffer(buffer);
        return this;
    }

    public SiliconKernel intVal(int value) {
        args.intVal(value);
        return this;
    }

    public SiliconKernel floatVal(float value) {
        args.floatVal(value);
        return this;
    }

    public SiliconKernel addLong(long value) {
        args.longVal(value);
        return this;
    }

    public SiliconKernel addDouble(double value) {
        args.doubleVal(value);
        return this;
    }

    public void launch(ComputeQueue queue, ComputeSize globalSize, ComputeSize localSize) {
        try {
            queue.dispatch(function, globalSize, localSize, args);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to launch kernel", e);
        }
    }

    public void launch(ComputeQueue queue, ComputeSize globalSize) {
        // this uses a default local work size of 256 for 1D
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

    public void launch(SiliconContext.QueueHandle queueHandle, ComputeSize globalSize, ComputeSize localSize) {
        launch(queueHandle.queue(), globalSize, localSize);
    }

    public void launchAndWait(ComputeQueue queue, ComputeSize globalSize, ComputeSize localSize) {
        try {
            queue.dispatch(function, globalSize, localSize, args);
            queue.awaitCompletion();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to launch kernel and wait", e);
        }
    }

    public ComputeArgs getArgs() {
        return args;
    }

    public ComputeFunction getFunction() {
        return function;
    }
}

