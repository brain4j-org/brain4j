package org.brain4j.math.gpu.silicon;

import org.silicon.api.Silicon;
import org.silicon.api.device.ComputeArena;
import org.silicon.api.device.ComputeBuffer;
import org.silicon.api.device.ComputeContext;
import org.silicon.api.device.ComputeDevice;
import org.silicon.api.kernel.ComputeQueue;

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
            this.device = Silicon.createDevice(deviceIndex);
            this.context = device.createContext();
            this.name = device.name();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create Silicon device at index " + deviceIndex, e);
        }
    }

    public SiliconDevice() {
        this(0);
    }

    public void createResources() {
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
    
    public ComputeBuffer createBuffer(float[] data, boolean persistent) {
        try {
            if (arena != null && !persistent) return arena.allocateArray(data);
            return context.allocateArray(data);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create buffer from float array", e);
        }
    }
    
    public ComputeBuffer createBuffer(int[] data, boolean persistent) {
        try {
            if (arena != null && !persistent) return arena.allocateArray(data);
            return context.allocateArray(data);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create buffer from int array", e);
        }
    }
    
    public ComputeBuffer createBuffer(long byteSize, boolean persistent) {
        try {
            if (arena != null && !persistent) return arena.allocateBytes(byteSize);
            return context.allocateBytes(byteSize);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create buffer of size " + byteSize, e);
        }
    }

    public ComputeBuffer createBuffer(float[] data) {
        return createBuffer(data, false);
    }

    public ComputeBuffer createBuffer(int[] data) {
        return createBuffer(data, false);
    }

    public ComputeBuffer createBuffer(long byteSize) {
        return createBuffer(byteSize, false);
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

    public void closeResources() {
        if (queue != null) {
            queue.await();
            queue = null;
        }
        
        if (arena != null) {
            arena.close();
            arena = null;
        }
    }
}

