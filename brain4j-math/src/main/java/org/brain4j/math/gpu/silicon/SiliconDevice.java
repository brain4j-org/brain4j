package org.brain4j.math.gpu.silicon;

import org.brain4j.math.tensor.TensorKey;
import org.silicon.api.Silicon;
import org.silicon.api.cache.MemoryPool;
import org.silicon.api.cache.Pooled;
import org.silicon.api.device.ComputeArena;
import org.silicon.api.device.ComputeBuffer;
import org.silicon.api.device.ComputeContext;
import org.silicon.api.device.ComputeDevice;
import org.silicon.api.kernel.ComputeQueue;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Supplier;

public class SiliconDevice {

    private final ComputeDevice device;
    private final ComputeContext context;
    private final int deviceIndex;
    private final String name;
    private final MemoryPool<TensorKey> memoryPool;
    private final List<Pooled> pooledQueue;
    
    private ComputeQueue queue;
    private ComputeArena arena;

    public SiliconDevice(int deviceIndex) {
        try {
            this.deviceIndex = deviceIndex;
            this.device = Silicon.createDevice(deviceIndex);
            this.context = device.createContext();
            this.name = device.name();
            this.memoryPool = context.createPool();
            this.pooledQueue = new ArrayList<>();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create Silicon device at index " + deviceIndex, e);
        }
    }

    public SiliconDevice() {
        this(0);
    }

    public void createResources() {
        if (!pooledQueue.isEmpty()) {
            pooledQueue.forEach(Pooled::close);
            pooledQueue.clear();
        }

        if (queue == null) {
            queue = context.createQueue();
        }
        
//        if (arena == null) {
//            arena = context.createArena();
//        }
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
    
    public ComputeBuffer acquire(TensorKey key, Supplier<ComputeBuffer> allocator) {
        Pooled pooled = memoryPool.acquire(key, allocator);
        pooledQueue.add(pooled);
        
        return pooled.value();
    }
    
    public ComputeBuffer acquire(TensorKey key, Supplier<ComputeBuffer> allocator, Consumer<ComputeBuffer> writer) {
        Pooled pooled = memoryPool.acquire(key, allocator);
        pooledQueue.add(pooled);

        ComputeBuffer result = pooled.value();
        writer.accept(result);

        return result;
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
    
    public MemoryPool<TensorKey> getMemoryPool() {
        return memoryPool;
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
        }
        
        if (arena != null) {
            arena.close();
            arena = null;
        }
        
        pooledQueue.forEach(Pooled::close); // frees this buffers to be used again
        pooledQueue.clear();
    }
    
    public ComputeBuffer copyBuffer(ComputeBuffer otherBuffer) {
        try {
            ComputeBuffer copy = arena != null
                ? arena.allocateBytes(otherBuffer.size())
                : context.allocateBytes(otherBuffer.size());
            
            otherBuffer.copyInto(copy);
            return copy;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create buffer from float array", e);
        }
    }
}

