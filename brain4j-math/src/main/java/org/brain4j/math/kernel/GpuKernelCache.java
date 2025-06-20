package org.brain4j.math.kernel;

import org.jocl.cl_kernel;
import org.jocl.cl_program;

import java.util.HashMap;
import java.util.Map;

import static org.jocl.CL.clCreateKernel;

public class GpuKernelCache {

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
}
