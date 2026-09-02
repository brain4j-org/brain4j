package org.brain4j.math.gpu.device;

import org.brain4j.math.tensor.TensorKey;
import org.silicon.api.Silicon;
import org.silicon.api.cache.MemoryPool;
import org.silicon.api.cache.Pooled;
import org.silicon.api.device.ComputeBuffer;
import org.silicon.api.device.ComputeContext;
import org.silicon.api.device.ComputeDevice;
import org.silicon.api.kernel.ComputeQueue;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * GPU device backed by the Silicon compute API.
 *
 * <p>The historical Brain4J type name is preserved while the implementation no
 * longer exposes raw native handles.
 */
public class Device {

    private final ComputeDevice device;
    private final ComputeContext context;
    private final MemoryPool<TensorKey> memoryPool;
    private final List<Pooled> pooledQueue;
    private final ComputeQueue queue;
    private final String name;
    private final int deviceIndex;
    private Thread threadContext;

    public Device() {
        this(0);
    }

    public Device(int deviceIndex) {
        try {
            this.deviceIndex = deviceIndex;
            this.device = Silicon.createDevice(deviceIndex);
            this.context = device.createContext();
            this.threadContext = Thread.currentThread();
            this.queue = context.createQueue();
            this.name = device.name();
            this.memoryPool = context.createPool();
            this.pooledQueue = new ArrayList<>();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create GPU device at index " + deviceIndex, e);
        }
    }

    private void ensureSameThread() {
        Thread current = Thread.currentThread();
        if (current != threadContext) {
            context.syncThread();
            threadContext = current;
        }
    }

    public synchronized void createResources() {
        ensureSameThread();

        if (!pooledQueue.isEmpty()) {
            pooledQueue.forEach(Pooled::close);
            pooledQueue.clear();
        }
    }

    public ComputeBuffer acquire(TensorKey key, Supplier<ComputeBuffer> allocator) {
        ensureSameThread();

        Pooled pooled = memoryPool.acquire(key, allocator);
        pooledQueue.add(pooled);

        return pooled.value();
    }

    public ComputeBuffer createBuffer(float[] data) {
        ensureSameThread();

        try {
            return context.allocateArray(data);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create buffer from float array", e);
        }
    }

    public ComputeBuffer createBuffer(int[] data) {
        ensureSameThread();

        try {
            return context.allocateArray(data);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create buffer from int array", e);
        }
    }

    public ComputeBuffer createBuffer(long byteSize) {
        ensureSameThread();

        try {
            return context.allocateBytes(byteSize);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create buffer of size " + byteSize, e);
        }
    }

    public MemoryPool<TensorKey> memoryPool() {
        return memoryPool;
    }

    public String name() {
        return name;
    }

    public int deviceIndex() {
        return deviceIndex;
    }

    public ComputeDevice device() {
        return device;
    }

    public ComputeContext context() {
        return context;
    }

    public ComputeQueue queue() {
        return queue;
    }

    public synchronized void free() {
        ensureSameThread();

        if (queue != null) {
            queue.await();
            queue.free();
        }

        memoryPool.free();
    }

    public synchronized void closeResources() {
        ensureSameThread();

        if (queue != null) {
            queue.await();
        }

        pooledQueue.forEach(Pooled::close);
        pooledQueue.clear();
    }

    @Override
    public String toString() {
        return "Device{" +
            "backend=Silicon" +
            ", name='" + name + '\'' +
            '}';
    }
}
