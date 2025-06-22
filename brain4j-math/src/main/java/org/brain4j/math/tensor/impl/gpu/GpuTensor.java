package org.brain4j.math.tensor.impl.gpu;

import org.brain4j.math.activation.Activation;
import org.brain4j.math.device.Device;
import org.brain4j.math.device.DeviceType;
import org.brain4j.math.device.DeviceUtils;
import org.brain4j.math.kernel.GpuContextHandler;
import org.brain4j.math.kernel.KernelFactory;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.TensorImplBase;
import org.brain4j.math.tensor.Tensors;
import org.brain4j.math.tensor.impl.cpu.CpuTensor;
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
    private static cl_kernel softmaxKernel = null;

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

        cl_command_queue queue = GpuContextHandler.queue();
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

        this.cleanable = CLEANER.register(this, new CollectableState(dataBuffer, shapeBuffer, stridesBuffer));
    }

    public cl_mem dataBuffer() {
        return dataBuffer;
    }

    public cl_mem stridesBuffer() {
        return stridesBuffer;
    }

    public cl_mem shapeBuffer() {
        return shapeBuffer;
    }

    public int size() {
        return size;
    }

    public static void initKernels(cl_context context) {
        cl_program tensorOpsProgram = DeviceUtils.createBuildProgram(context, "/kernels/basic/tensor_ops.cl");
        cl_program elementaryOpsProgram = DeviceUtils.createBuildProgram(context, "/kernels/basic/elementary_ops.cl");

        matmulKernel = clCreateKernel(tensorOpsProgram, "matmul", null);
        addKernel = clCreateKernel(tensorOpsProgram, "add", null);
        subKernel = clCreateKernel(tensorOpsProgram, "sub", null);
        mulKernel = clCreateKernel(tensorOpsProgram, "mul", null);
        divKernel = clCreateKernel(tensorOpsProgram, "div", null);

        transposeKernel = clCreateKernel(tensorOpsProgram, "transpose", null);
        sumAlongDimKernel = clCreateKernel(tensorOpsProgram, "sum_along_dim", null);
        layerNormKernel = clCreateKernel(tensorOpsProgram, "layer_norm", null);
        softmaxKernel = clCreateKernel(tensorOpsProgram, "softmax_last_dim", null);

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
        cl_command_queue queue = GpuContextHandler.queue();

        KernelFactory
            .create(kernel)
            .addMemParam(dataBuffer)
            .addFloatParam(value)
            .addIntParam(size)
            .launch(queue, 1, size);

        return this;
    }

    private Tensor launchElementaryKernel(cl_kernel kernel, Tensor other) {
        if (!(other instanceof GpuTensor)) {
            other = other.gpu();
        }

        GpuTensor B = (GpuTensor) other;

        int broadcastDim = (Arrays.equals(shape, B.shape)) ? -1 : shape[1];
        int batch = (broadcastDim == -1) ? 0 : shape[0];

        cl_command_queue queue = GpuContextHandler.queue();

        KernelFactory
            .create(kernel)
            .addMemParam(dataBuffer)
            .addMemParam(B.dataBuffer)
            .addIntParam(size)
            .addIntParam(broadcastDim)
            .addIntParam(batch)
            .launch(queue, 1, size);

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

        int inRowStride = strides[0];
        int inColStride = strides[1];
        int outRowStride = result.strides[0];
        int outColStride = result.strides[1];

        cl_command_queue queue = GpuContextHandler.queue();

        KernelFactory
            .create(transposeKernel)
            .addMemParam(dataBuffer)
            .addMemParam(result.dataBuffer)
            .addIntParam(rows)
            .addIntParam(cols)
            .addIntParam(inRowStride)
            .addIntParam(inColStride)
            .addIntParam(outRowStride)
            .addIntParam(outColStride)
            .launch(queue, 2, rows, cols);

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
        cl_command_queue queue = GpuContextHandler.queue();

        KernelFactory
            .create(sqrtKernel)
            .addMemParam(dataBuffer)
            .addIntParam(size)
            .launch(queue, 1, size);

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
        GpuTensor result = new GpuTensor(outShape);

        cl_command_queue queue = GpuContextHandler.queue();

        int TILE_SIZE = 16;

        long[] globalWorkSize = new long[] {roundUp(TILE_SIZE, M), roundUp(TILE_SIZE, P)};
        long[] localWorkSize = new long[] {TILE_SIZE, TILE_SIZE};

        KernelFactory
            .create(matmulKernel)
            .addMemParam(dataBuffer)
            .addMemParam(B.dataBuffer)
            .addMemParam(result.dataBuffer)
            .addIntParam(M)
            .addIntParam(K)
            .addIntParam(P)
            .launch(queue, 2, globalWorkSize, localWorkSize);

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

        cl_command_queue queue = GpuContextHandler.queue();
        GpuTensor result = new GpuTensor(newShape);

        KernelFactory
            .create(sumAlongDimKernel)
            .addMemParam(dataBuffer)
            .addMemParam(result.dataBuffer)
            .addIntParam(reducedSize)
            .addIntParam(outerSize)
            .addIntParam(innerSize)
            .launch(queue, 2, outerSize, innerSize);

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

        cl_command_queue queue = GpuContextHandler.queue();

        KernelFactory
            .create(layerNormKernel)
            .addMemParam(dataBuffer)
            .addIntParam(batchSize)
            .addIntParam(featuresSize)
            .addFloatParam((float) epsilon)
            .launch(queue, 1, batchSize);

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
    public int elements() {
        return size;
    }

    @Override
    public Tensor softmax() {
        return super.softmax();
    }

    @Override
    public Tensor softmax(double temperature) {
        cl_command_queue queue = GpuContextHandler.queue();

        GpuTensor result = new GpuTensor(shape);

        int lastDim = shape[shape.length - 1];
        int rows = size / lastDim;

        KernelFactory
            .create(softmaxKernel)
            .addMemParam(dataBuffer)
            .addMemParam(result.dataBuffer)
            .addIntParam(lastDim)
            .addFloatParam((float) temperature)
            .launch(queue, 1, rows);

        return result;
    }

    @Override
    public void release() {
        cleanable.clean();
    }
}
