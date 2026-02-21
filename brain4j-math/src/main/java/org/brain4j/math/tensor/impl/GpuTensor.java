package org.brain4j.math.tensor.impl;

import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.device.DeviceUtils;
import org.brain4j.math.gpu.kernel.KernelFactory;
import org.brain4j.math.gpu.memory.GpuQueue;
import org.brain4j.math.gpu.memory.TempBuffer;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.commons.Range;

import java.util.Arrays;

import static org.lwjgl.opencl.CL10.*;

public class GpuTensor extends BaseTensor {

    private final Device device;
    private final TempBuffer shapeBuffer;
    private final TempBuffer stridesBuffer;
    private final int size;
    private TempBuffer dataBuffer;

    public GpuTensor(Device device, int[] shape, float... data) {
        this.device = device;
        this.size = Tensors.computeSize(shape);
        this.shape = shape;
        this.strides = Tensors.computeStrides(shape);

        long readFlag = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
        long flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;

        if (data.length == 0) {
            data = new float[size];
        }

        this.shapeBuffer = device.createBuffer(readFlag, shape);
        this.stridesBuffer = device.createBuffer(readFlag, strides);
        this.dataBuffer = device.createBuffer(flags, data);
    }

    public GpuTensor(Device device, int[] shape, long otherBuffer) {
        this.device = device;
        this.size = Tensors.computeSize(shape);
        this.shape = shape;
        this.strides = Tensors.computeStrides(shape);

        long dataSize = (long) Float.BYTES * this.size;
        long readFlag = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;

        this.shapeBuffer = device.createBuffer(readFlag, shape);
        this.stridesBuffer = device.createBuffer(readFlag, strides);
        this.dataBuffer = device.createBuffer(CL_MEM_READ_WRITE, dataSize);
        
        try (GpuQueue queue = GpuContext.getOrCreate(device)) {
            clEnqueueCopyBuffer(queue.pointer(), otherBuffer, dataBuffer.value(), 0, 0, dataSize,
                null, null);
        }
    }

    public GpuTensor(Device device, int[] shape, int[] strides, float... data) {
        this.device = device;
        this.size = data.length == 0 ? Tensors.computeSize(shape) : data.length;
        this.shape = shape;
        this.strides = strides;

        long dataSize  = (long) Float.BYTES * this.size;
        long readFlag = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
        long flags = CL_MEM_READ_WRITE;

        if (data.length > 0) {
            flags |= CL_MEM_COPY_HOST_PTR;
        }

        TempBuffer dataBuffer = data.length == 0
            ? device.createBuffer(flags, dataSize)
            : device.createBuffer(flags, data);

        this.shapeBuffer = device.createBuffer(readFlag, shape);
        this.stridesBuffer = device.createBuffer(readFlag, strides);
        this.dataBuffer = dataBuffer;
    }

    public GpuTensor(GpuTensor reference, int[] newShape) {
        this.device = reference.device;
        this.size = Tensors.computeSize(newShape);
        this.shape = newShape;
        this.strides = Tensors.computeStrides(newShape);

        long readFlag = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
        
        // make sure GC does not close this resource
        reference.dataBuffer.retain();
        
        this.shapeBuffer = device.createBuffer(readFlag, shape);
        this.stridesBuffer = device.createBuffer(readFlag, strides);
        this.dataBuffer = reference.dataBuffer;
    }

    public Device device() {
        return device;
    }

    public long dataBuffer() {
        return dataBuffer.value();
    }

    public long stridesBuffer() {
        return stridesBuffer.value();
    }

    public long shapeBuffer() {
        return shapeBuffer.value();
    }

    public int size() {
        return size;
    }

    public static void initKernels(Device device) {
        long tensorOpsProgram = DeviceUtils.createBuildProgram(device, "/kernels/basic/tensor_ops.cl");
        long elementaryOpsProgram = DeviceUtils.createBuildProgram(device, "/kernels/basic/elementary_ops.cl");
        long activationsProgram = DeviceUtils.createBuildProgram(device, "/kernels/basic/activations.cl");
        long gradientClipProgram = DeviceUtils.createBuildProgram(device, "/kernels/basic/gradient_clippers.cl");
        long attentionProgram = DeviceUtils.createBuildProgram(device, "/kernels/attention/flash_attention.cl");
        
        String[] tensorOpsKernels = { "slice", "concat_last_dim", "concat_copy_a", "concat_copy_b", "matmul_batched",
            "add", "sub", "mul", "div", "sum_along_dim", "softmax_last_dim", "layer_norm" };

        for (String kernel : tensorOpsKernels) {
            GpuContext.register(device, kernel, tensorOpsProgram);
        }

        String[] scalarKernels = { "mask", "add_scalar", "mul_scalar", "div_scalar", "pow_scalar", "sqrt" };

        for (String kernel : scalarKernels) {
            GpuContext.register(device, kernel, elementaryOpsProgram);
        }
        
        for (Activations activation : Activations.values()) {
            Activation function = activation.function();
            String prefix = function.getKernelPrefix();
            
            GpuContext.register(device, prefix + "_forward", activationsProgram);
            GpuContext.register(device, prefix + "_backward", activationsProgram);
        }
        
        GpuContext.register(device, "hard_clip", gradientClipProgram);
        GpuContext.register(device, "l2_clip", gradientClipProgram);
        GpuContext.register(device, "flash_attention_forward", attentionProgram);
        GpuContext.register(device, "flash_attention_forward_with_lse", attentionProgram);
        GpuContext.register(device, "flash_attention_backward", attentionProgram);
        GpuContext.register(device, "flash_attention_backward_dq", attentionProgram);
        GpuContext.register(device, "flash_attention_forward_tiled", attentionProgram);
        GpuContext.register(device, "flash_attention_backward_tiled", attentionProgram);
    }

    private long roundUp(int globalSize) {
        int r = globalSize % 16;
        if (r == 0) return globalSize;
        return globalSize + 16 - r;
    }

    private Tensor launchScalarKernel(String kernelName, float value) {
        try (GpuQueue queue = GpuContext.getOrCreate(device)) {
            KernelFactory.create(device, kernelName)
                .addMemParam(dataBuffer)
                .addFloatParam(value)
                .addIntParam(size)
                .launch(queue, 1, size);
        }

        return this;
    }

    private Tensor launchElementaryKernel(String kernelName, Tensor other) {
        if (!(other instanceof GpuTensor)) {
            other = other.to(device);
        }

        GpuTensor B = (GpuTensor) other;

        int broadcastDim = (Arrays.equals(shape, B.shape)) ? -1 : shape[1];
        int batch = (broadcastDim == -1) ? 0 : shape[0];

        try (GpuQueue queue = GpuContext.getOrCreate(device)) {
            KernelFactory.create(device, kernelName)
                .addMemParam(dataBuffer)
                .addMemParam(B.dataBuffer)
                .addIntParam(size)
                .addIntParam(broadcastDim)
                .addIntParam(batch)
                .launch(queue, 1, size);
        }

        return this;
    }

    @Override
    public Tensor to(Object device) {
        if (device == null) {
            Tensor result = new CpuTensor(Shape.of(shape), data());
            result.setAutogradContext(autogradContext);
            return result;
        }

        return this;
    }

    @Override
    public Tensor transpose(int dim1, int dim2) {
        int rank = shape.length;

        if (rank == 1) {
            return reshape(1, elements());
        }

        int[] newShape = shape.clone();
        newShape[dim1] = shape[dim2];
        newShape[dim2] = shape[dim1];

        int[] newStrides = strides.clone();
        newStrides[dim2] = strides[dim1];
        newStrides[dim1] = strides[dim2];

        GpuTensor view = new GpuTensor(device, newShape, newStrides);

        view.dataBuffer = dataBuffer;
        view.transposed = !transposed;

        return view;
    }

    @Override
    public Tensor add(Tensor other) {
        return launchElementaryKernel("add", other);
    }

    @Override
    public Tensor add(double value) {
        return launchScalarKernel("add_scalar", (float) value);
    }

    @Override
    public Tensor sub(Tensor other) {
        return launchElementaryKernel("sub", other);
    }

    @Override
    public Tensor sub(double value) {
        return launchScalarKernel("sub_scalar", (float) value);
    }

    @Override
    public Tensor mul(Tensor other) {
        return launchElementaryKernel("mul", other);
    }

    @Override
    public Tensor mul(double value) {
        return launchScalarKernel("mul_scalar", (float) value);
    }

    @Override
    public Tensor div(Tensor other) {
        return launchElementaryKernel("div", other);
    }

    @Override
    public Tensor div(double value) {
        return launchScalarKernel("div_scalar", (float) value);
    }

    @Override
    public Tensor pow(double value) {
        return launchScalarKernel("pow_scalar", (float) value);
    }
    
    @Override
    public Tensor pow(Tensor other) {
        return null;
    }
    
    @Override
    public Tensor sqrt() {
        try (GpuQueue queue = GpuContext.getOrCreate(device)) {
            KernelFactory.create(device, "sqrt")
                .addMemParam(dataBuffer)
                .addIntParam(size)
                .launch(queue, 1, size);
        }

        return this;
    }

    @Override
    public Tensor activate(Activation activation) {
        return super.activate(activation);
    }
    
    @Override
    public Tensor matmul(Tensor other) {
        if (!(other instanceof GpuTensor B)) {
            return matmul(other.to(device));
        }
        
        int[] shapeA = shape();
        int[] shapeB = other.shape();
        
        if (shapeA.length < 2 || shapeB.length < 2) {
            throw new IllegalArgumentException("Both tensors must have rank >= 2.");
        }
        
        int M = shapeA[shapeA.length - 2];
        int K = shapeA[shapeA.length - 1];
        int Kb = shapeB[shapeB.length - 2];
        int P = shapeB[shapeB.length - 1];
        
        if (K != Kb) {
            throw new IllegalArgumentException("Incompatible inner dims for matmul: K != Kb (" + K + " != " + Kb + ")");
        }
        
        int aBatchRank = shapeA.length - 2;
        int bBatchRank = shapeB.length - 2;
        int maxBatchRank = Math.max(aBatchRank, bBatchRank);
        
        int[] aBatch = new int[maxBatchRank];
        int[] bBatch = new int[maxBatchRank];
        for (int i = 0; i < maxBatchRank; ++i) {
            int ai = i - (maxBatchRank - aBatchRank);
            int bi = i - (maxBatchRank - bBatchRank);
            aBatch[i] = (ai >= 0) ? shapeA[ai] : 1;
            bBatch[i] = (bi >= 0) ? shapeB[bi] : 1;
        }
        
        int[] outBatch = new int[maxBatchRank];
        long batchCountLong = 1;
        for (int i = 0; i < maxBatchRank; ++i) {
            int da = aBatch[i];
            int db = bBatch[i];
            
            if (da == db || da == 1 || db == 1) {
                outBatch[i] = Math.max(da, db);
            } else {
                throw new IllegalArgumentException("Cannot broadcast batch dimension: " + Arrays.toString(aBatch) +
                    " vs " + Arrays.toString(bBatch));
            }
            
            batchCountLong *= outBatch[i];
            
            if (batchCountLong > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Batch size too large");
            }
        }
        int batchCount = (int) batchCountLong;
        
        int[] outShape = new int[maxBatchRank + 2];
        System.arraycopy(outBatch, 0, outShape, 0, maxBatchRank);
        
        outShape[maxBatchRank] = M;
        outShape[maxBatchRank + 1] = P;
        
        GpuTensor result = new GpuTensor(device, outShape);
        
        int matrixSizeA = M * K;
        int matrixSizeB = K * P;
        int matrixSizeC = M * P;
        
        int[] outStrides = new int[maxBatchRank];
        int a = 1;
        
        for (int i = maxBatchRank - 1; i >= 0; --i) {
            outStrides[i] = a;
            a *= outBatch[i];
        }
        
        int[] offsetsA = new int[batchCount];
        int[] offsetsB = new int[batchCount];
        int[] offsetsC = new int[batchCount];
        
        for (int b = 0; b < batchCount; ++b) {
            int[] idx = new int[maxBatchRank];
            for (int i = 0; i < maxBatchRank; ++i) {
                idx[i] = (b / outStrides[i]) % outBatch[i];
            }
            
            int linearA = 0;
            for (int i = 0; i < aBatchRank; ++i) {
                int alignedPos = i + (maxBatchRank - aBatchRank);
                int dimSizeA = shapeA[i];
                int chosen = (dimSizeA == 1) ? 0 : idx[alignedPos];
                linearA = linearA * dimSizeA + chosen;
            }
            
            int linearB = 0;
            for (int i = 0; i < bBatchRank; ++i) {
                int alignedPos = i + (maxBatchRank - bBatchRank);
                int dimSizeB = shapeB[i];
                int chosen = (dimSizeB == 1) ? 0 : idx[alignedPos];
                linearB = linearB * dimSizeB + chosen;
            }
            
            offsetsA[b] = linearA * matrixSizeA;
            offsetsB[b] = linearB * matrixSizeB;
            offsetsC[b] = b * matrixSizeC;
        }
        
        final int TILE_SIZE = 16;
        long[] globalWorkSize = new long[] {
            roundUp(M),
            roundUp(P),
            batchCount
        };
        long[] localWorkSize = new long[] { TILE_SIZE, TILE_SIZE, 1 };
        long flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;

        TempBuffer memoryA = device.createBuffer(flags, offsetsA);
        TempBuffer memoryB = device.createBuffer(flags, offsetsB);
        TempBuffer memoryC = device.createBuffer(flags, offsetsC);

        try (GpuQueue queue = GpuContext.getOrCreate(device)) {
            KernelFactory.create(device, "matmul_batched")
                .addMemParam(dataBuffer)
                .addMemParam(B.dataBuffer)
                .addMemParam(result.dataBuffer)
                .addMemParam(memoryA)
                .addMemParam(memoryB)
                .addMemParam(memoryC)
                .addIntParam(M)
                .addIntParam(K)
                .addIntParam(P)
                .addIntParam(batchCount)
                .addIntParam(transposed ? 1 : 0)
                .addIntParam(other.transposed() ? 1 : 0)
                .launch(queue, 3, globalWorkSize, localWorkSize);
        }
        
        memoryA.release();
        memoryB.release();
        memoryC.release();
        
        return result;
    }
    
    @Override
    public Tensor sum(int dim, boolean keepDim) {
        if (dim < 0 || dim >= shape.length) {
            throw new IllegalArgumentException("Dimension " + dim + " out of bounds for tensor of shape " + Arrays.toString(shape));
        }

        int[] newShape = Tensors.computeNewShape(shape, dim, keepDim);
        int reducedSize = shape[dim];

        int outerSize = 1;
        for (int i = 0; i < dim; i++) outerSize *= shape[i];

        int innerSize = 1;
        for (int i = dim + 1; i < shape.length; i++) innerSize *= shape[i];
        
        GpuTensor result = new GpuTensor(device, newShape);
        result.setAutogradContext(autogradContext);
        
        try (GpuQueue queue = GpuContext.getOrCreate(device)) {
            KernelFactory.create(device, "sum_along_dim")
                .addMemParam(dataBuffer)
                .addMemParam(result.dataBuffer)
                .addIntParam(outerSize)
                .addIntParam(reducedSize)
                .addIntParam(innerSize)
                .launch(queue, 2, outerSize, innerSize);
        }

        return result;
    }

    @Override
    public Tensor reshape(int... newShape) {
        int newSize = Tensors.computeSize(newShape);

        if (newSize != size) {
            throw new IllegalArgumentException(
                "The total new dimension (" + newSize + ") does not match the current dimension (" + data().length + ")"
            );
        }

        return new GpuTensor(this, newShape);
    }

    @Override
    public Tensor concat(Tensor other) {
        if (!(other instanceof GpuTensor)) {
            other = other.to(device);
        }

        GpuTensor B = (GpuTensor) other;

        if (shape.length != B.shape.length) {
            throw new IllegalArgumentException("Concatenation is only supported for tensors with the same number of dimensions.");
        }
        for (int i = 0; i < shape.length - 1; i++) {
            if (shape[i] != B.shape[i]) {
                throw new IllegalArgumentException("Shapes must match on all dimensions except the last.");
            }
        }

        int rank = shape.length;
        int lastA = shape[rank - 1];
        int lastB = B.shape[rank - 1];
        int concatLast = lastA + lastB;

        int[] newShape = Arrays.copyOf(shape, rank);
        newShape[rank - 1] = concatLast;

        int outerSize = 1;
        for (int i = 0; i < rank - 1; i++) outerSize *= shape[i];

        GpuTensor result = new GpuTensor(device, newShape);

        try (GpuQueue queue = GpuContext.getOrCreate(device)) {
            KernelFactory.create(device, "concat_last_dim")
                .addMemParam(this.dataBuffer)
                .addMemParam(B.dataBuffer)
                .addMemParam(result.dataBuffer)
                .addIntParam(outerSize)
                .addIntParam(lastA)
                .addIntParam(lastB)
                .addIntParam(concatLast)
                .launch(queue, 1, (long) outerSize * concatLast);
        }

        return result;
    }

    @Override
    public Tensor concat(Tensor other, int dimension) {
        if (!(other instanceof GpuTensor)) {
            other = other.to(device);
        }

        GpuTensor B = (GpuTensor) other;

        if (shape.length != B.shape.length) {
            throw new IllegalArgumentException("Tensors must have the same rank.");
        }

        int rank = rank();
        if (dimension < 0 || dimension >= rank) {
            throw new IllegalArgumentException("Invalid dimension: " + dimension);
        }

        for (int i = 0; i < rank; i++) {
            if (i != dimension && shape[i] != B.shape[i]) {
                throw new IllegalArgumentException("Shapes must match in all dimensions except the concatenation one.");
            }
        }

        int[] newShape = Arrays.copyOf(shape, rank);
        newShape[dimension] += B.shape[dimension];

        int blockSize = 1;
        int numBlocks = 1;
        for (int i = dimension + 1; i < rank; i++) blockSize *= shape[i];
        for (int i = 0; i < dimension; i++) numBlocks *= shape[i];

        int thisDim = shape[dimension];
        int otherDim = B.shape[dimension];

        GpuTensor result = new GpuTensor(device, newShape);

        int totalA = numBlocks * thisDim * blockSize;
        int totalB = numBlocks * otherDim * blockSize;

        try (GpuQueue queue = GpuContext.getOrCreate(device)) {
            KernelFactory.create(device, "concat_copy_a")
                .addMemParam(this.dataBuffer)
                .addMemParam(result.dataBuffer)
                .addIntParam(numBlocks)
                .addIntParam(thisDim)
                .addIntParam(otherDim)
                .addIntParam(blockSize)
                .launch(queue, 1, totalA);

            KernelFactory.create(device, "concat_copy_b")
                .addMemParam(B.dataBuffer)
                .addMemParam(result.dataBuffer)
                .addIntParam(numBlocks)
                .addIntParam(thisDim)
                .addIntParam(otherDim)
                .addIntParam(blockSize)
                .launch(queue, 1, totalB);
        }

        return result;
    }

    @Override
    public Tensor slice(Range... ranges) {
        if (ranges.length > shape.length) {
            throw new IllegalArgumentException("Too many ranges specified");
        }

        int[] newShape = new int[shape.length];

        for (int i = 0; i < shape.length; i++) {
            if (i < ranges.length && ranges[i] != null) {
                newShape[i] = ranges[i].size(shape[i]);
            } else {
                newShape[i] = shape[i];
            }
        }

        GpuTensor result = new GpuTensor(device, newShape);

        int[] starts = new int[ranges.length];
        int[] steps = new int[ranges.length];

        for (int i = 0; i < ranges.length; i++) {
            Range range = ranges[i];
            starts[i] = range == null ? 0 : range.start();
            steps[i] = range == null ? 1 : range.step();
        }

        long flags = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;

        TempBuffer memoryShape = device.createBuffer(flags, newShape);
        TempBuffer memoryStart = device.createBuffer(flags, starts);
        TempBuffer memoryStep = device.createBuffer(flags, steps);

        try (GpuQueue queue = GpuContext.getOrCreate(device)) {
            KernelFactory.create(device, "slice")
                .addMemParam(this.dataBuffer)
                .addMemParam(result.dataBuffer)
                .addMemParam(this.stridesBuffer)
                .addMemParam(result.stridesBuffer)
                .addMemParam(memoryShape)
                .addMemParam(memoryStart)
                .addMemParam(memoryStep)
                .addIntParam(rank())
                .launch(queue, 1, result.elements());
        }
        
        memoryShape.release();
        memoryStart.release();
        memoryStep.release();

        return result;
    }

    @Override
    public Tensor layerNorm(double epsilon) {
        GpuTensor result = new GpuTensor(device, shape);
        result.setAutogradContext(autogradContext);

        int rank = shape.length;
        int featuresSize = shape[rank - 1];
        int batchSize = 1;

        for (int i = 0; i < rank - 1; i++) batchSize *= shape[i];

        try (GpuQueue queue = GpuContext.getOrCreate(device)) {
            KernelFactory.create(device, "layer_norm")
                .addMemParam(this.dataBuffer)
                .addMemParam(result.dataBuffer)
                .addIntParam(featuresSize)
                .addIntParam(batchSize)
                .addFloatParam((float) epsilon)
                .launch(queue, 1, batchSize);
        }

        return result;
    }

    @Override
    public float[] data() {
        float[] buffer = new float[size];
        long queue = device.newCommandQueue();

        try {
            clEnqueueReadBuffer(queue, dataBuffer.value(), true, 0, buffer, null, null);
        } finally {
            clFinish(queue);
            clReleaseCommandQueue(queue);
        }

        return buffer;
    }

    @Override
    public Tensor set(float value, int... indices) {
        return null;
    }

    @Override
    public int elements() {
        return size;
    }

    @Override
    public Tensor softmax(double temperature) {
        GpuTensor result = new GpuTensor(device, shape);

        int lastDim = shape[shape.length - 1];
        int rows = size / lastDim;

        try (GpuQueue queue = GpuContext.getOrCreate(device)) {
            KernelFactory.create(device, "softmax_last_dim")
                .addMemParam(dataBuffer)
                .addMemParam(result.dataBuffer)
                .addIntParam(lastDim)
                .addFloatParam((float) temperature)
                .launch(queue, 1, rows);
        }

        return result;
    }
    
    @Override
    public Tensor copy() {
        return new GpuTensor(device, shape, dataBuffer.value());
    }
}
