package org.brain4j.math.gpu.silicon;

import org.silicon.Silicon;
import org.silicon.computing.ComputeQueue;
import org.silicon.device.ComputeBuffer;
import org.silicon.device.ComputeContext;
import org.silicon.device.ComputeDevice;

public class SiliconDevice {

    private final ComputeDevice device;
    private final ComputeContext context;
    private ComputeQueue queue;
    private final int deviceIndex;
    private final String name;

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
                queue = context.createQueue();
            } catch (Throwable e) {
                throw new RuntimeException("Failed to create compute queue", e);
            }
        }
    }

    public ComputeQueue newQueue() {
        try {
            return context.createQueue();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create compute queue", e);
        }
    }

    public SiliconBuffer createBuffer(float[] data) {
        try {
            ComputeBuffer buffer = context.allocateArray(data);
            return new SiliconBuffer(buffer);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create buffer from float array", e);
        }
    }

    public SiliconBuffer createBuffer(int[] data) {
        try {
            ComputeBuffer buffer = context.allocateArray(data);
            return new SiliconBuffer(buffer);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create buffer from int array", e);
        }
    }

    public SiliconBuffer createBuffer(long byteSize) {
        try {
            ComputeBuffer buffer = context.allocateBytes(byteSize);
            return new SiliconBuffer(buffer);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create buffer of size " + byteSize, e);
        }
    }

    public SiliconBuffer createBufferAsync(
            float[] data,
            ComputeQueue queue
    ) {
        try {
            ComputeBuffer buffer = context.allocateArray(data, queue);
            return new SiliconBuffer(buffer);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create buffer from float array (async)", e);
        }
    }

    public SiliconBuffer createBufferAsync(
            int[] data,
            ComputeQueue queue
    ) {
        try {
            ComputeBuffer buffer = context.allocateArray(data, queue);
            return new SiliconBuffer(buffer);
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
        if (queue == null) {
            createQueue();
        }
        return queue;
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

    @Override
    public String toString() {
        return "SiliconDevice{" +
            "name='" + name + '\'' +
            ", deviceIndex=" + deviceIndex +
            '}';
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        SiliconDevice other = (SiliconDevice) obj;
        return deviceIndex == other.deviceIndex;
    }

    @Override
    public int hashCode() {
        return Integer.hashCode(deviceIndex);
    }
}

