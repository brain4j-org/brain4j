package org.brain4j.math.gpu.memory;

import org.silicon.api.kernel.ComputeQueue;

public final class GpuQueue implements AutoCloseable {

    private final ComputeQueue queue;
    private final boolean temporary;

    public GpuQueue(ComputeQueue queue, boolean temporary) {
        if (queue == null) {
            throw new IllegalArgumentException("Compute queue must not be null");
        }
        this.queue = queue;
        this.temporary = temporary;
    }

    public ComputeQueue queue() {
        return queue;
    }

    public boolean temporary() {
        return temporary;
    }

    @Deprecated
    public long pointer() {
        throw new UnsupportedOperationException(
            "Raw native command queue handles are not available in the Silicon GPU backend"
        );
    }

    @Override
    public void close() {
        if (!temporary) return;

        queue.await();
        queue.free();
    }
}
