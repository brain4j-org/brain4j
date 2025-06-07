package org.brain4j.math.tensor.gpu;

import org.jocl.cl_mem;

import java.util.concurrent.atomic.AtomicBoolean;

import static org.jocl.CL.clReleaseMemObject;

public class CollectableState implements Runnable {

    private final AtomicBoolean released = new AtomicBoolean(false);
    private final cl_mem shapeBuf;
    private final cl_mem stridesBuf;
    private final cl_mem dataBuf;

    public CollectableState(cl_mem dataBuf, cl_mem shapeBuf, cl_mem stridesBuf) {
        this.dataBuf = dataBuf;
        this.shapeBuf = shapeBuf;
        this.stridesBuf = stridesBuf;
    }

    @Override
    public void run() {
        if (released.compareAndSet(false, true)) {
            clReleaseMemObject(shapeBuf);
            clReleaseMemObject(stridesBuf);
            clReleaseMemObject(dataBuf);
        }
    }
}
