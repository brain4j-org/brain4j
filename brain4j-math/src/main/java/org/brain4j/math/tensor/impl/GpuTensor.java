package org.brain4j.math.tensor.impl;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.device.Device;
import org.brain4j.math.device.DeviceType;
import org.brain4j.math.device.DeviceUtils;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.TensorImplBase;
import org.brain4j.math.tensor.Tensors;
import org.brain4j.math.tensor.gpu.CollectableState;
import org.jocl.*;

import java.lang.ref.Cleaner;
import java.util.Arrays;

import static org.jocl.CL.*;

public class GpuTensor extends TensorImplBase {

    private static final Cleaner CLEANER = Cleaner.create();

    private static cl_kernel matmulKernel = null;
    private static cl_kernel addKernel = null;
    private static cl_kernel subKernel = null;
    private static cl_kernel mulKernel = null;
    private static cl_kernel divKernel = null;
    private static cl_kernel transposeKernel = null;
    private static cl_kernel sumAlongDimKernel = null;
    private static cl_kernel layerNormKernel = null;

    private static cl_kernel addScalarKernel = null;
    private static cl_kernel subScalarKernel = null;
    private static cl_kernel mulScalarKernel = null;
    private static cl_kernel divScalarKernel = null;
    private static cl_kernel powScalarKernel = null;
    private static cl_kernel sqrtKernel = null;

    private final Cleaner.Cleanable cleanable;
    private final cl_mem shapeBuffer;
    private final cl_mem stridesBuffer;
    private final cl_mem dataBuffer;
    private final int size;

    public GpuTensor(int[] shape, float... data) {
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

        this.cleanable = CLEANER.register(this, new CollectableState(dataBuffer, shapeBuffer, stridesBuffer));
    }

    public GpuTensor(int[] shape, cl_mem otherBuffer) {
        this.size = computeSize(shape);
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

        this.dataBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, dataSize, null, null);

        cl_command_queue queue = device.newCommandQueue();
        clEnqueueCopyBuffer(
                queue,
                otherBuffer,
                this.dataBuffer,
                0,
                0,
                dataSize,
                0,
                null,
                null
        );

        clFinish(queue);
        clReleaseCommandQueue(queue);

        this.cleanable = CLEANER.register(this, new CollectableState(dataBuffer, shapeBuffer, stridesBuffer));
    }

    public static void initKernels(cl_context context) {
        String tensorOpsSource = DeviceUtils.readKernelSource("/kernels/basic/tensor_ops.cl");
        String elementaryOpsSource = DeviceUtils.readKernelSource("/kernels/basic/elementary_ops.cl");

        cl_program tensorOpsProgram = clCreateProgramWithSource(context, 1, new String[]{tensorOpsSource}, null, null);
        cl_program elementaryOpsProgram = clCreateProgramWithSource(context, 1, new String[]{elementaryOpsSource}, null, null);

        clBuildProgram(tensorOpsProgram, 0, null, null, null, null);
        clBuildProgram(elementaryOpsProgram, 0, null, null, null, null);

        matmulKernel = clCreateKernel(tensorOpsProgram, "matmul", null);
        addKernel = clCreateKernel(tensorOpsProgram, "add", null);
        subKernel = clCreateKernel(tensorOpsProgram, "sub", null);
        mulKernel = clCreateKernel(tensorOpsProgram, "mul", null);
        divKernel = clCreateKernel(tensorOpsProgram, "div", null);

        transposeKernel = clCreateKernel(tensorOpsProgram, "transpose", null);
        sumAlongDimKernel = clCreateKernel(tensorOpsProgram, "sum_along_dim", null);
        layerNormKernel = clCreateKernel(tensorOpsProgram, "layer_norm", null);

        addScalarKernel = clCreateKernel(elementaryOpsProgram, "add_scalar", null);
        subScalarKernel = clCreateKernel(elementaryOpsProgram, "sub_scalar", null);
        mulScalarKernel = clCreateKernel(elementaryOpsProgram, "mul_scalar", null);
        divScalarKernel = clCreateKernel(elementaryOpsProgram, "div_scalar", null);
        powScalarKernel = clCreateKernel(elementaryOpsProgram, "pow_scalar", null);
        sqrtKernel = clCreateKernel(elementaryOpsProgram, "sqrt", null);
    }

    private long roundUp(int groupSize, int globalSize) {
        int r = globalSize % groupSize;
        if (r == 0) return globalSize;
        return globalSize + groupSize - r;
    }

    private Tensor launchScalarKernel(cl_kernel kernel, float value) {
        clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(dataBuffer));
        clSetKernelArg(kernel, 1, Sizeof.cl_float, Pointer.to(new float[]{value}));
        clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{size}));

        long[] globalWorkSize = new long[] { size };

        Device device = DeviceUtils.currentDevice();

        cl_command_queue queue = device.newCommandQueue();
        clEnqueueNDRangeKernel(
                queue,
                kernel,
                1,
                null,
                globalWorkSize,
                null,
                0,
                null,
                null
        );

        clFinish(queue);
        clReleaseCommandQueue(queue);

        return this;
    }

    private Tensor launchElementaryKernel(cl_kernel kernel, Tensor other) {
        if (!(other instanceof GpuTensor)) {
            other = other.gpu();
        }

        GpuTensor B = (GpuTensor) other;

        int broadcastDim = (Arrays.equals(shape, B.shape)) ? -1 : shape[1];
        int batch = (broadcastDim == -1) ? 0 : shape[0];

        Pointer aPtr = Pointer.to(this.dataBuffer);
        Pointer bPtr = Pointer.to(B.dataBuffer);

        Device device = DeviceUtils.currentDevice();
        cl_command_queue queue = device.newCommandQueue();

        clSetKernelArg(kernel, 0, Sizeof.cl_mem, aPtr);
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, bPtr);
        clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{size}));
        clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{broadcastDim}));
        clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{batch}));

        clEnqueueNDRangeKernel(
                queue,
                kernel,
                1,
                null,
                new long[]{size},
                null,
                0,
                null,
                null
        );

        clFinish(queue);
        clReleaseCommandQueue(queue);

        return this;
    }

    @Override
    public Tensor clone() {
        return new GpuTensor(shape, this.dataBuffer);
    }

    @Override
    public Tensor to(DeviceType deviceType) {
        return switch (deviceType) {
            case CPU -> new CpuTensor(shape, data());
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

        GpuTensor result = Tensors.matrix(cols, rows).gpu();

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

        clEnqueueNDRangeKernel(
                queue,
                transposeKernel,
                2,
                null,
                globalWorkSize,
                null,
                0,
                null,
                null
        );

        clFinish(queue);
        clReleaseCommandQueue(queue);

        return result;
    }

    @Override
    public Tensor add(Tensor other) {
        return launchElementaryKernel(addKernel, other);
    }

    @Override
    public Tensor add(double value) {
        return launchScalarKernel(addScalarKernel, (float) value);
    }

    @Override
    public Tensor sub(Tensor other) {
        return launchElementaryKernel(subKernel, other);
    }

    @Override
    public Tensor sub(double value) {
        return launchScalarKernel(subScalarKernel, (float) value);
    }

    @Override
    public Tensor mul(Tensor other) {
        return launchElementaryKernel(mulKernel, other);
    }

    @Override
    public Tensor mul(double value) {
        return launchScalarKernel(mulScalarKernel, (float) value);
    }

    @Override
    public Tensor div(Tensor other) {
        return launchElementaryKernel(divKernel, other);
    }

    @Override
    public Tensor div(double value) {
        return launchScalarKernel(divScalarKernel, (float) value);
    }

    @Override
    public Tensor pow(double value) {
        return launchScalarKernel(powScalarKernel, (float) value);
    }

    @Override
    public Tensor sqrt() {
        clSetKernelArg(sqrtKernel, 0, Sizeof.cl_mem, Pointer.to(dataBuffer));
        clSetKernelArg(sqrtKernel, 1, Sizeof.cl_int, Pointer.to(new int[]{size}));

        long[] globalWorkSize = new long[] { size };

        Device device = DeviceUtils.currentDevice();

        cl_command_queue queue = device.newCommandQueue();
        clEnqueueNDRangeKernel(
            queue,
            sqrtKernel,
            1,
            null,
            globalWorkSize,
            null,
            0,
            null,
            null
        );

        clFinish(queue);
        clReleaseCommandQueue(queue);

        return this;
    }

    @Override
    public Tensor activate(Activation activation) {
        return super.activate(activation);
    }

    @Override
    public Tensor matmul(Tensor other) {
        if (!(other instanceof GpuTensor B)) {
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
        GpuTensor result = new GpuTensor(outShape, dummy);

        Device device = DeviceUtils.currentDevice();
        cl_command_queue queue = device.newCommandQueue();

        clSetKernelArg(matmulKernel, 0, Sizeof.cl_mem, Pointer.to(this.dataBuffer));
        clSetKernelArg(matmulKernel, 1, Sizeof.cl_mem, Pointer.to(B.dataBuffer));
        clSetKernelArg(matmulKernel, 2, Sizeof.cl_mem, Pointer.to(result.dataBuffer));
        clSetKernelArg(matmulKernel, 3, Sizeof.cl_int, Pointer.to(new int[] { M }));
        clSetKernelArg(matmulKernel, 4, Sizeof.cl_int, Pointer.to(new int[] { K }));
        clSetKernelArg(matmulKernel, 5, Sizeof.cl_int, Pointer.to(new int[] { P }));

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
        clReleaseCommandQueue(queue);

        return result;
    }

    @Override
    public Tensor sum(int dim, boolean keepDim) {
        if (dim < 0 || dim >= shape.length) {
            throw new IllegalArgumentException("Dimension " + dim + " out of bounds for tensor of shape " + Arrays.toString(shape));
        }

        int[] newShape = computeNewShape(shape, dim, keepDim);
        int reducedSize = shape[dim];

        int outerSize = 1;
        for (int i = 0; i < dim; i++) outerSize *= shape[i];

        int innerSize = 1;
        for (int i = dim + 1; i < shape.length; i++) innerSize *= shape[i];

        Device device = DeviceUtils.currentDevice();
        cl_command_queue queue = device.newCommandQueue();

        float[] dummy = new float[0];
        GpuTensor result = new GpuTensor(newShape, dummy);

        clSetKernelArg(sumAlongDimKernel, 0, Sizeof.cl_mem, Pointer.to(dataBuffer));
        clSetKernelArg(sumAlongDimKernel, 1, Sizeof.cl_mem, Pointer.to(result.dataBuffer));
        clSetKernelArg(sumAlongDimKernel, 2, Sizeof.cl_int, Pointer.to(new int[] {outerSize}));
        clSetKernelArg(sumAlongDimKernel, 3, Sizeof.cl_int, Pointer.to(new int[] {reducedSize}));
        clSetKernelArg(sumAlongDimKernel, 4, Sizeof.cl_int, Pointer.to(new int[] {innerSize}));

        long[] globalWorkSize = new long[] {outerSize, innerSize};
        clEnqueueNDRangeKernel(
                queue,
                sumAlongDimKernel,
                2,
                null,
                globalWorkSize,
                null,
                0,
                null,
                null
        );

        clFinish(queue);
        clReleaseCommandQueue(queue);

        return result;
    }

    @Override
    public Tensor layerNorm(double epsilon) {
        int batchSize = 1;
        int featuresSize = shape[0];

        if (shape.length == 2) {
            batchSize = shape[0];
            featuresSize = shape[1];
        }

        Device device = DeviceUtils.currentDevice();
        cl_command_queue queue = device.newCommandQueue();

        clSetKernelArg(layerNormKernel, 0, Sizeof.cl_mem, Pointer.to(dataBuffer));
        clSetKernelArg(layerNormKernel, 1, Sizeof.cl_int, Pointer.to(new int[]{batchSize}));
        clSetKernelArg(layerNormKernel, 2, Sizeof.cl_int, Pointer.to(new int[]{featuresSize}));
        clSetKernelArg(layerNormKernel, 3, Sizeof.cl_float, Pointer.to(new float[]{(float) epsilon}));

        long[] globalWorkSize = new long[] { batchSize };

        clEnqueueNDRangeKernel(
                queue,
                layerNormKernel,
                1,
                null,
                globalWorkSize,
                null,
                0,
                null,
                null
        );

        clFinish(queue);
        clReleaseCommandQueue(queue);

        return this;
    }

    @Override
    public float[] data() {
        float[] buffer = new float[size];

        Device device = DeviceUtils.currentDevice();
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

        clFinish(queue);
        clReleaseCommandQueue(queue);

        return buffer;
    }

    @Override
    public void release() {
        cleanable.clean();
    }
}
