package org.brain4j.math.gpu.device;

import org.brain4j.math.gpu.memory.GpuQueue;
import org.brain4j.math.gpu.memory.TempBuffer;
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
    private final int deviceIndex;
    private final String name;
    private final MemoryPool<TensorKey> memoryPool;
    private final List<Pooled> pooledQueue;

    private ComputeQueue queue;
    private GpuQueue legacyQueue;
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
            this.legacyQueue = new GpuQueue(queue, false);
            this.name = device.name();
            this.memoryPool = context.createPool();
            this.pooledQueue = new ArrayList<>();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to create GPU device at index " + deviceIndex, e);
        }
    }

    /**
     * Legacy raw-handle constructor retained for source compatibility.
     *
     * <p>Raw native handles are not part of the Silicon backend. The values are
     * ignored and the default device is selected.
     */
    @Deprecated
    public Device(long platformAddr, long deviceAddr) {
        this(0);
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

    public TempBuffer createBuffer(long flags, float[] data) {
        return new TempBuffer(createBuffer(data));
    }

    public TempBuffer createBuffer(long flags, int[] data) {
        return new TempBuffer(createBuffer(data));
    }

    public TempBuffer createBuffer(long flags, long dataSize) {
        return new TempBuffer(createBuffer(dataSize));
    }

    public void createQueue() {
        setQueue(context.createQueue());
    }

    public MemoryPool<TensorKey> getMemoryPool() {
        return memoryPool;
    }

    public String getName() {
        return name;
    }

    public String name() {
        return name;
    }

    public int getDeviceIndex() {
        return deviceIndex;
    }

    public ComputeDevice computeDevice() {
        return device;
    }

    public ComputeContext context() {
        return context;
    }

    public ComputeQueue queue() {
        return queue;
    }

    public void setQueue(ComputeQueue queue) {
        this.queue = queue;
        this.legacyQueue = queue == null ? null : new GpuQueue(queue, false);
    }

    public GpuQueue getQueue() {
        return legacyQueue;
    }

    public void setQueue(GpuQueue queue) {
        this.legacyQueue = queue;
        this.queue = queue == null ? null : queue.queue();
    }

    public void printLimits() {
        System.out.println("Device = " + device.name());
        System.out.println("Vendor = " + device.vendor());
        System.out.println("Memory = " + device.memorySize());
    }

    public synchronized void free() {
        ensureSameThread();

        if (queue != null) {
            queue.await();
            queue.free();
            queue = null;
            legacyQueue = null;
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

    @Deprecated
    public long getPlatform() {
        throw unsupportedRawHandle("platform");
    }

    @Deprecated
    public long getDevice() {
        throw unsupportedRawHandle("device");
    }

    @Deprecated
    public long getContext() {
        throw unsupportedRawHandle("context");
    }

    @Deprecated
    public long newContext() {
        throw unsupportedRawHandle("context");
    }

    @Deprecated
    public long newCommandQueue() {
        throw unsupportedRawHandle("command queue");
    }

    private static UnsupportedOperationException unsupportedRawHandle(String handle) {
        return new UnsupportedOperationException(
            "Raw native " + handle + " handles are not available in the Silicon GPU backend"
        );
    }

    @Override
    public String toString() {
        return "Device{" +
            "backend=Silicon" +
            ", name='" + name + '\'' +
            '}';
    }
}
