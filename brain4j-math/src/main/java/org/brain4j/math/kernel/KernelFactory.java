package org.brain4j.math.kernel;

import org.jocl.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clSetKernelArg;

public class KernelFactory {

    public record Argument(int index, int size, Pointer pointer) { }

    private final cl_kernel kernel;
    private final List<Argument> arguments;

    protected KernelFactory(cl_kernel kernel) {
        this.kernel = kernel;
        this.arguments = new ArrayList<>();
    }

    public static KernelFactory create(cl_kernel kernel) {
        return new KernelFactory(kernel);
    }

    public KernelFactory addIntParam(int variable) {
        arguments.add(new Argument(arguments.size(), Sizeof.cl_int, Pointer.to(new int[]{variable})));
        return this;
    }

    public KernelFactory addFloatParam(float variable) {
        arguments.add(new Argument(arguments.size(), Sizeof.cl_float, Pointer.to(new float[]{variable})));
        return this;
    }

    public KernelFactory addMemParam(cl_mem memory) {
        arguments.add(new Argument(arguments.size(), Sizeof.cl_mem, Pointer.to(memory)));
        return this;
    }

    public void run(cl_command_queue queue, int workDim, long... globalWorkSize) {
        for (Argument argument : arguments) {
            clSetKernelArg(kernel, argument.index, argument.size, argument.pointer);
        }

        clEnqueueNDRangeKernel(queue, kernel, workDim, null, globalWorkSize, null,
            0, null, null);
    }

    public void run(cl_command_queue queue, int workDim, long[] globalWorkSize, long... localWorkSize) {
        for (Argument argument : arguments) {
            clSetKernelArg(kernel, argument.index, argument.size, argument.pointer);
        }

        clEnqueueNDRangeKernel(queue, kernel, workDim, null, globalWorkSize, localWorkSize,
            0, null, null);
    }
}
