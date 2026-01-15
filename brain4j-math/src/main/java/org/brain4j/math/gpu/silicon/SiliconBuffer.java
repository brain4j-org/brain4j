package org.brain4j.math.gpu.silicon;

import org.silicon.device.ComputeBuffer;
import org.silicon.computing.ComputeQueue;

import java.lang.ref.Cleaner;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

public class SiliconBuffer {

    private static final Cleaner CLEANER = Cleaner.create();

    private final AtomicInteger refCount = new AtomicInteger(1);
    private ComputeBuffer buffer;
    private final Cleaner.Cleanable cleanable;

    public SiliconBuffer(ComputeBuffer buffer) {
        this.buffer = buffer;
        this.cleanable = CLEANER.register(this, new CleanerTask(buffer, refCount));
    }

    public ComputeBuffer getBuffer() {
        return buffer;
    }

    public ComputeBuffer setBuffer(ComputeBuffer buffer) {
        this.buffer = buffer;
        return buffer;
    }

    public void retain() {
        refCount.incrementAndGet();
    }

    public void release() {
        int count = refCount.decrementAndGet();
        if (count <= 0) {
            try {
                buffer.free();
            } catch (Throwable e) {
                // errors are ignored for now
            }
        }
    }

    /**
     * Gets the current reference count.
     *
     * @return the reference count
     */
    public int getRefCount() {
        return refCount.get();
    }

    public SiliconBuffer copy() {
        try {
            ComputeBuffer copyBuffer = buffer.copy();
            return new SiliconBuffer(copyBuffer);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to copy buffer", e);
        }
    }

    public SiliconBuffer copyInto(SiliconBuffer other) {
        try {
            buffer.copyInto(other.buffer);
            return this;
        } catch (Throwable e) {
            throw new RuntimeException("Failed to copy buffer into another", e);
        }
    }

    public SiliconBuffer copyAsync(ComputeQueue queue) {
        try {
            ComputeBuffer copyBuffer = buffer.copyAsync(queue);
            return new SiliconBuffer(copyBuffer);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to copy buffer async", e);
        }
    }

    public void get(float[] data) {
        try {
            buffer.get(data);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get float data from buffer", e);
        }
    }

    public void get(int[] data) {
        try {
            buffer.get(data);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to get int data from buffer", e);
        }
    }

    private static class CleanerTask implements Runnable {
        private final AtomicBoolean released = new AtomicBoolean(false);
        private final ComputeBuffer buffer;
        private final AtomicInteger refCount;

        CleanerTask(ComputeBuffer buffer, AtomicInteger refCount) {
            this.buffer = buffer;
            this.refCount = refCount;
        }

        @Override
        public void run() {
            if (released.compareAndSet(false, true) && refCount.get() > 0) {
                try {
                    buffer.free();
                } catch (Throwable e) {
                    // also do the same here
                }
            }
        }
    }
}

