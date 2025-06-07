package org.brain4j.math.tensor.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.device.Device;
import org.brain4j.math.device.DeviceType;
import org.brain4j.math.device.DeviceUtils;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;
import org.jocl.*;

import java.util.Arrays;

import static org.jocl.CL.*;

public class TensorGPU extends TensorImplBase {

    private static cl_program program = null;
    private static cl_kernel matmulKernel = null;
    private static cl_kernel addKernel = null;
    private static cl_kernel mulKernel = null;
    private static cl_kernel transposeKernel = null;

    private final cl_mem shapeBuffer;
    private final cl_mem stridesBuffer;
    private final cl_mem dataBuffer;

    private final int size;

    public TensorGPU(int[] shape, float... data) {
        this.size = data.length == 0 ? computeSize(shape) : data.length;
        this.shape = shape;
        this.strides = computeStrides(shape);

        Device device = DeviceUtils.currentDevice();
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

    public static synchronized void initKernels(cl_context context) {
        if (program != null && matmulKernel != null) {
            return;
        }

        String source = DeviceUtils.readKernelSource("/kernels/basic/tensor_ops.cl");

        program = clCreateProgramWithSource(context, 1, new String[]{ source }, null, null);
        clBuildProgram(program, 0, null, null, null, null);

        matmulKernel = clCreateKernel(program, "matmul", null);
        addKernel = clCreateKernel(program, "add", null);
        mulKernel = clCreateKernel(program, "mul", null);
        transposeKernel = clCreateKernel(program, "transpose", null);

        if (matmulKernel == null || addKernel == null || mulKernel == null || transposeKernel == null) {
            throw new RuntimeException("Failed to create kernel.");
        }
    }

    private long roundUp(int groupSize, int globalSize) {
        int r = globalSize % groupSize;
        if (r == 0) return globalSize;
        return globalSize + groupSize - r;
    }

    @Override
    public Tensor clone() {
        return new TensorGPU(shape, data());
    }

    @Override
    public Tensor to(DeviceType deviceType) {
        return switch (deviceType) {
            case CPU -> new TensorCPU(shape, data);
            case GPU -> this;
            default -> throw new IllegalArgumentException("Unsupported device type: " + deviceType);
        };
    }

    @Override
    public Tensor transpose() {
        if (dimension() == 1) {
            return reshape(1, elements());
        }

        if (shape.length != 2) {
            throw new UnsupportedOperationException(
                "transpose() is supported only for 2D tensors, not for tensors with " + shape.length + " dimensions"
            );
        }

        int rows = shape[0];
        int cols = shape[1];

        TensorGPU result = Tensors.matrix(cols, rows).gpu();

        if (usesGrad()) {
            result.setAutogradContext(autogradContext);
        }

        Pointer aPtr = Pointer.to(this.dataBuffer);
        Pointer bPtr = Pointer.to(result.dataBuffer);

        int inRowStride = strides[0];
        int inColStride = strides[1];
        int outRowStride = result.strides[0];
        int outColStride = result.strides[1];

        clSetKernelArg(transposeKernel, 0, Sizeof.cl_mem, aPtr);
        clSetKernelArg(transposeKernel, 1, Sizeof.cl_mem, bPtr);
        clSetKernelArg(transposeKernel, 2, Sizeof.cl_int, Pointer.to(new int[]{rows}));
        clSetKernelArg(transposeKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{cols}));
        clSetKernelArg(transposeKernel, 4, Sizeof.cl_int, Pointer.to(new int[]{inRowStride}));
        clSetKernelArg(transposeKernel, 5, Sizeof.cl_int, Pointer.to(new int[]{inColStride}));
        clSetKernelArg(transposeKernel, 6, Sizeof.cl_int, Pointer.to(new int[]{outRowStride}));
        clSetKernelArg(transposeKernel, 7, Sizeof.cl_int, Pointer.to(new int[]{outColStride}));

        Device device = DeviceUtils.currentDevice();
        cl_command_queue queue = device.newCommandQueue();

        long[] globalWorkSize = new long[]{ rows, cols };

        clEnqueueNDRangeKernel(queue, transposeKernel, 2, null, globalWorkSize,
            null, 0, null, null
        );
        clFinish(queue);

        return result;
    }

    @Override
    public Tensor add(Tensor other) {
        if (!(other instanceof TensorGPU B)) {
            throw new IllegalArgumentException("Other tensor is not an instance of TensorGPU.");
        }

        int broadcastDim = (Arrays.equals(shape, B.shape)) ? -1 : shape[1];
        int batch = (broadcastDim == -1) ? 0 : shape[0];

        Pointer aPtr = Pointer.to(this.dataBuffer);
        Pointer bPtr = Pointer.to(B.dataBuffer);

        Device device = DeviceUtils.currentDevice();
        cl_command_queue queue = device.newCommandQueue();

        clSetKernelArg(addKernel, 0, Sizeof.cl_mem, aPtr);
        clSetKernelArg(addKernel, 1, Sizeof.cl_mem, bPtr);
        clSetKernelArg(addKernel, 2, Sizeof.cl_int, Pointer.to(new int[]{size}));
        clSetKernelArg(addKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{broadcastDim}));
        clSetKernelArg(addKernel, 4, Sizeof.cl_int, Pointer.to(new int[]{batch}));

        clEnqueueNDRangeKernel(queue, addKernel, 1, null, new long[]{size}, null,
            0, null, null);
        clFinish(queue);

        return this;
    }

    @Override
    public Tensor mul(Tensor other) {
        if (!(other instanceof TensorGPU B)) {
            throw new IllegalArgumentException("Other tensor is not an instance of TensorGPU.");
        }

        int broadcastDim = (Arrays.equals(shape, B.shape)) ? -1 : shape[1];
        int batch = (broadcastDim == -1) ? 0 : shape[0];

        Pointer aPtr = Pointer.to(this.dataBuffer);
        Pointer bPtr = Pointer.to(B.dataBuffer);

        Device device = DeviceUtils.currentDevice();
        cl_command_queue queue = device.newCommandQueue();

        clSetKernelArg(mulKernel, 0, Sizeof.cl_mem, aPtr);
        clSetKernelArg(mulKernel, 1, Sizeof.cl_mem, bPtr);
        clSetKernelArg(mulKernel, 2, Sizeof.cl_int, Pointer.to(new int[]{size}));
        clSetKernelArg(mulKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{broadcastDim}));
        clSetKernelArg(mulKernel, 4, Sizeof.cl_int, Pointer.to(new int[]{batch}));

        clEnqueueNDRangeKernel(queue, mulKernel, 1, null, new long[]{size}, null,
            0, null, null);

        return this;
    }

    @Override
    public Tensor activate(Activation activation) {
        return super.activate(activation);
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
        TensorGPU result = new TensorGPU(outShape, dummy);

        Device device = DeviceUtils.currentDevice();
        cl_command_queue queue = device.newCommandQueue();

        int arg = 0;
        clSetKernelArg(matmulKernel, arg++, Sizeof.cl_mem, Pointer.to(this.dataBuffer));
        clSetKernelArg(matmulKernel, arg++, Sizeof.cl_mem, Pointer.to(B.dataBuffer));
        clSetKernelArg(matmulKernel, arg++, Sizeof.cl_mem, Pointer.to(result.dataBuffer));
        clSetKernelArg(matmulKernel, arg++, Sizeof.cl_int, Pointer.to(new int[] { M }));
        clSetKernelArg(matmulKernel, arg++, Sizeof.cl_int, Pointer.to(new int[] { K }));
        clSetKernelArg(matmulKernel, arg, Sizeof.cl_int, Pointer.to(new int[] { P }));

        final int TILE_SIZE = 16;

        long[] globalWorkSize = new long[] {
                roundUp(TILE_SIZE, M),
                roundUp(TILE_SIZE, P)
        };

        long[] localWorkSize = new long[] {
                TILE_SIZE,
                TILE_SIZE
        };

        clEnqueueNDRangeKernel(queue, matmulKernel, 2, null, globalWorkSize, localWorkSize,
                0, null, null);
        clFinish(queue);

        return result;
    }

    @Override
    public float[] data() {
        float[] buffer = new float[size];

        Device device = DeviceUtils.currentDevice();
        cl_command_queue queue = device.newCommandQueue();

        clEnqueueReadBuffer(queue, dataBuffer, CL_TRUE, 0, (long) size * Sizeof.cl_float, Pointer.to(buffer),
                0, null, null);
        clFinish(queue);

        return buffer;
    }

    public void release() {
        clReleaseMemObject(shapeBuffer);
        clReleaseMemObject(stridesBuffer);
        clReleaseMemObject(dataBuffer);
    }
}
