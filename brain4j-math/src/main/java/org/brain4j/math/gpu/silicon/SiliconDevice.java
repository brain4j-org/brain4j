package org.brain4j.math.gpu.silicon;

import org.silicon.Silicon;
import org.silicon.computing.ComputeArgs;
import org.silicon.computing.ComputeQueue;
import org.silicon.device.ComputeArena;
import org.silicon.device.ComputeBuffer;
import org.silicon.device.ComputeContext;
import org.silicon.device.ComputeDevice;

public class SiliconDevice {

    private final ComputeDevice device;
    private final ComputeContext context;
    private final int deviceIndex;
    private final String name;
    
    private ComputeQueue queue;
    private ComputeArena arena;

    public SiliconDevice(int deviceIndex) {
        try {
            this.deviceIndex = deviceIndex;
            this.device = Silicon.createSystemDevice(deviceIndex);
            this.context = device.createContext();
            this.name = device.getName();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create Silicon device at index " + deviceIndex, e);
        }
    }

    public SiliconDevice() {
        this(0);
    }

    public void createQueue() {
        if (queue == null) {
            try {
                this.queue = context.createQueue();
                this.arena = context.createArena();
            } catch (Throwable e) {
                throw new RuntimeException("Failed to create compute queue", e);
            }
        }
    }

    public ComputeQueue newTempQueue() {
        try {
            return context.createQueue();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create compute queue", e);
        }
    }
    
    public ComputeArena newTempArena() {
        try {
            return context.createArena();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create compute arena", e);
        }
    }

    public ComputeBuffer createBuffer(float[] data) {
        try {
            if (arena != null) return arena.allocateArray(data);
            return context.allocateArray(data);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create buffer from float array", e);
        }
    }

    public ComputeBuffer createBuffer(int[] data) {
        try {
            if (arena != null) return arena.allocateArray(data);
            return context.allocateArray(data);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create buffer from int array", e);
        }
    }

    public ComputeBuffer createBuffer(long byteSize) {
        try {
            if (arena != null) return arena.allocateBytes(byteSize);
            return context.allocateBytes(byteSize);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create buffer of size " + byteSize, e);
        }
    }

    public ComputeBuffer createBufferAsync(float[] data, ComputeQueue queue) {
        try {
            return context.allocateArray(data, queue);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create buffer from float array (async)", e);
        }
    }

    public ComputeBuffer createBufferAsync(int[] data, ComputeQueue queue) {
        try {
            return context.allocateArray(data, queue);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create buffer from int array (async)", e);
        }
    }

    public String getName() {
        return name;
    }

    public int getDeviceIndex() {
        return deviceIndex;
    }

    public ComputeDevice getDevice() {
        return device;
    }

    public ComputeContext getContext() {
        return context;
    }

    public ComputeQueue getQueue() {
        return queue;
    }
    
    public ComputeArena getArena() {
        return arena;
    }
    
    public void setQueue(ComputeQueue queue) {
        this.queue = queue;
    }

    public void releaseQueue() {
        if (queue != null) {
            try {
                queue.awaitCompletion();
                queue.release();
            } catch (Throwable ignored) {}
            queue = null;
        }
    }
}

