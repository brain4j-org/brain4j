package org.brain4j.math.kernel;

import org.jocl.cl_command_queue;
import org.jocl.cl_kernel;
import org.jocl.cl_program;

import java.util.HashMap;
import java.util.Map;

import static org.jocl.CL.*;

public class GpuContextHandler {

    private static final ThreadLocal<cl_command_queue> queue = new ThreadLocal<>();
    private static final Map<String, cl_kernel> kernels = new HashMap<>();

    public static void register(String kernelName, cl_program program) {
        if (kernels.containsKey(kernelName)) {
            throw new IllegalArgumentException("Kernel " + kernelName + " already initialized!");
        }

        cl_kernel kernel = clCreateKernel(program, kernelName, null);
        kernels.put(kernelName, kernel);
    }

    public static cl_kernel kernel(String kernelName) {
        return kernels.get(kernelName);
    }

    public static void updateQueue(cl_command_queue queue) {
        GpuContextHandler.queue.set(queue);
    }

    public static cl_command_queue queue() {
        return queue.get();
    }

    public static void closeQueue(cl_command_queue queue) {
        clFinish(queue);
        clReleaseCommandQueue(queue);
    }

    public static void closeQueue() {
        cl_command_queue queue = GpuContextHandler.queue.get();
        clFinish(queue);
        clReleaseCommandQueue(queue);
    }
}
