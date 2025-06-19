package org.brain4j.math.tensor.gpu;

import org.jocl.cl_command_queue;

import static org.jocl.CL.clFinish;
import static org.jocl.CL.clReleaseCommandQueue;

public class OpenCLContext {

    private static final ThreadLocal<cl_command_queue> currentQueue = new ThreadLocal<>();

    public static void updateQueue(cl_command_queue queue) {
        currentQueue.set(queue);
    }

    public static cl_command_queue currentQueue() {
        return currentQueue.get();
    }

    public static void closeQueue() {
        cl_command_queue queue = currentQueue.get();
        clFinish(queue);
        clReleaseCommandQueue(queue);
    }
}
