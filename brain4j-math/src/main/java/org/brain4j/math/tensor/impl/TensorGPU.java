package org.brain4j.math.tensor.impl;

import org.brain4j.math.device.Device;
import org.brain4j.math.device.DeviceUtils;
import org.brain4j.math.tensor.Tensor;
import org.jocl.*;

import java.util.Arrays;

import static org.jocl.CL.*;

public class TensorGPU extends TensorImplBase {

    private static cl_program program = null;
    private static cl_kernel kernelMatMul2D = null;

    private final cl_mem shapeBuffer;
    private final cl_mem stridesBuffer;
    private final cl_mem dataBuffer;

    private final int size;

    public TensorGPU(int[] shape, float... data) {
        this.size = data.length == 0 ? computeSize(shape) : data.length;
        this.shape = shape;
        this.strides = computeStrides(shape);

        Device device = DeviceUtils.device();
        cl_context context = device.context();

        long shapeSize = (long) Sizeof.cl_int * shape.length;
        long stridesSize = (long) Sizeof.cl_int * strides.length;
        long dataSize  = (long) Sizeof.cl_float * this.size;

        this.shapeBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                shapeSize, Pointer.to(shape), null);
        this.stridesBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                stridesSize, Pointer.to(strides), null);

        if (data.length > 0) {
            this.dataBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                    dataSize, Pointer.to(data), null);
        } else {
            this.dataBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, dataSize, null, null);
        }
    }

    @Override
    public Tensor add(Tensor other) {
        return super.add(other);
    }

    @Override
    public Tensor mul(Tensor other) {
        return super.mul(other);
    }

    @Override
    public Tensor matmul(Tensor other) {
        if (!(other instanceof TensorGPU B)) {
            throw new IllegalArgumentException("Other tensor is not an instance of TensorGPU.");
        }

        int[] shapeA = shape();
        int[] shapeB = other.shape();

        if (shapeA[1] != shapeB[0]) {
            throw new IllegalArgumentException("Incompatible shapes for matrix multiplication: " +
                    Arrays.toString(shapeA) + " and " + Arrays.toString(shapeB));
        }

        int M = shapeA[0];
        int K = shapeA[1];
        int P = shapeB[1];

        int[] outShape = new int[]{ M, P };
        float[] dummy = new float[0];
        TensorGPU C = new TensorGPU(outShape, dummy);

        Device device = DeviceUtils.device();
        cl_command_queue queue = device.newCommandQueue();

        int arg = 0;
        clSetKernelArg(kernelMatMul2D, arg++, Sizeof.cl_mem, Pointer.to(this.dataBuffer));
        clSetKernelArg(kernelMatMul2D, arg++, Sizeof.cl_mem, Pointer.to(B.dataBuffer));
        clSetKernelArg(kernelMatMul2D, arg++, Sizeof.cl_mem, Pointer.to(C.dataBuffer));
        clSetKernelArg(kernelMatMul2D, arg++, Sizeof.cl_int, Pointer.to(new int[] { M }));
        clSetKernelArg(kernelMatMul2D, arg++, Sizeof.cl_int, Pointer.to(new int[] { K }));
        clSetKernelArg(kernelMatMul2D, arg, Sizeof.cl_int, Pointer.to(new int[] { P }));

        long[] globalWorkSize = new long[]{ (long) M, (long) P };

        clEnqueueNDRangeKernel(
                queue,
                kernelMatMul2D,
                2,
                null,
                globalWorkSize,
                null,
                0,
                null,
                null
        );

        clFinish(queue);
        return C;
    }

    @Override
    public float[] data() {
        float[] buffer = new float[size];

        Device device = DeviceUtils.device();
        cl_command_queue queue = device.newCommandQueue();

        clEnqueueReadBuffer(
                queue,
                dataBuffer,
                CL_TRUE,
                0,
                (long) size * Sizeof.cl_float,
                Pointer.to(buffer),
                0,
                null,
                null
        );

        return buffer;
    }

    public void release() {
        clReleaseMemObject(shapeBuffer);
        clReleaseMemObject(stridesBuffer);
        clReleaseMemObject(dataBuffer);
    }

    public static synchronized void initKernels(cl_context context) {
        if (program != null && kernelMatMul2D != null) {
            return;
        }

        String source = DeviceUtils.readKernelSource("/kernels/basic/tensor_ops.cl");

        program = clCreateProgramWithSource(context, 1, new String[]{ source }, null, null);
        clBuildProgram(program, 0, null, null, null, null);

        kernelMatMul2D = clCreateKernel(program, "matmul", null);
    }
}
