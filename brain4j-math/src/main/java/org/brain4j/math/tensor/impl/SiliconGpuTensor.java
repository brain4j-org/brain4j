package org.brain4j.math.tensor.impl;

import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.activation.Activations;
import org.brain4j.math.gpu.silicon.SiliconBuffer;
import org.brain4j.math.gpu.silicon.SiliconContext;
import org.brain4j.math.gpu.silicon.SiliconDevice;
import org.brain4j.math.gpu.silicon.SiliconKernel;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.index.Range;
import org.silicon.computing.ComputeQueue;
import org.silicon.computing.ComputeSize;
import org.silicon.kernel.ComputeModule;

import java.util.Arrays;

import static java.util.Arrays.stream;

public class SiliconGpuTensor extends BaseTensor {

    private static final int TILE_SIZE = 16;

    private final SiliconDevice device;
    private final SiliconBuffer shapeBuffer;
    private final SiliconBuffer stridesBuffer;
    private final int size;
    private SiliconBuffer dataBuffer;

    public SiliconGpuTensor(SiliconDevice device, int[] shape, float... data) {
        this.device = device;
        this.size = Tensors.computeSize(shape);
        this.shape = shape;
        this.strides = Tensors.computeStrides(shape);

        if (data.length == 0) {
            data = new float[size];
        }

        this.shapeBuffer = device.createBuffer(shape);
        this.stridesBuffer = device.createBuffer(strides);
        this.dataBuffer = device.createBuffer(data);
    }

    public SiliconGpuTensor(SiliconDevice device, int[] shape, SiliconBuffer otherBuffer) {
        this.device = device;
        this.size = Tensors.computeSize(shape);
        this.shape = shape;
        this.strides = Tensors.computeStrides(shape);

        this.shapeBuffer = device.createBuffer(shape);
        this.stridesBuffer = device.createBuffer(strides);
        this.dataBuffer = otherBuffer.copy();
    }

    public SiliconGpuTensor(SiliconDevice device, int[] shape, int[] strides, float... data) {
        this.device = device;
        this.size = data.length == 0 ? Tensors.computeSize(shape) : data.length;
        this.shape = shape;
        this.strides = strides;

        if (data.length > 0) {
            this.dataBuffer = device.createBuffer(data);
        } else {
            long dataSize = (long) Float.BYTES * this.size;
            this.dataBuffer = device.createBuffer(dataSize);
        }

        this.shapeBuffer = device.createBuffer(shape);
        this.stridesBuffer = device.createBuffer(strides);
    }

    public SiliconGpuTensor(SiliconGpuTensor reference, int[] newShape) {
        this.device = reference.device;
        this.size = Tensors.computeSize(newShape);
        this.shape = newShape;
        this.strides = Tensors.computeStrides(newShape);

        // let's share the data buffer (to retain the reference)
        reference.dataBuffer.retain();

        this.shapeBuffer = device.createBuffer(shape);
        this.stridesBuffer = device.createBuffer(strides);
        this.dataBuffer = reference.dataBuffer;
    }

    public SiliconDevice device() {
        return device;
    }

    public SiliconBuffer dataBuffer() {
        return dataBuffer;
    }

    public SiliconBuffer shapeBuffer() {
        return shapeBuffer;
    }

    public SiliconBuffer stridesBuffer() {
        return stridesBuffer;
    }

    public int size() {
        return size;
    }

    public static void initKernels(SiliconDevice device) {
        try {
            ComputeModule tensorOpsModule = device.getContext().loadModuleFromResources("/shaders/tensor_ops.slang");
            ComputeModule elementaryOpsModule = device.getContext().loadModuleFromResources("/shaders/elementary_ops.slang");
            ComputeModule activationsModule = device.getContext().loadModuleFromResources("/shaders/activations.slang");
            ComputeModule gradientClipModule = device.getContext().loadModuleFromResources("/shaders/gradient_clippers.slang");
            ComputeModule flashAttentionModule = device.getContext().loadModuleFromResources("/shaders/flash_attention.slang");
            ComputeModule fftModule = device.getContext().loadModuleFromResources("/shaders/fft.slang");
            ComputeModule convolutionModule = device.getContext().loadModuleFromResources("/shaders/convolution.slang");
            ComputeModule complexOpsModule = device.getContext().loadModuleFromResources("/shaders/complex_ops.slang");

            SiliconContext.storeModule(device, "tensor_ops", tensorOpsModule);
            SiliconContext.storeModule(device, "elementary_ops", elementaryOpsModule);
            SiliconContext.storeModule(device, "activations", activationsModule);
            SiliconContext.storeModule(device, "gradient_clippers", gradientClipModule);
            SiliconContext.storeModule(device, "flash_attention", flashAttentionModule);
            SiliconContext.storeModule(device, "fft", fftModule);
            SiliconContext.storeModule(device, "convolution", convolutionModule);
            SiliconContext.storeModule(device, "complex_ops", complexOpsModule);

            String[] tensorOpsKernels = {
                "slice", "concat_last_dim", "concat_copy_a", "concat_copy_b",
                "matmul_batched", "matmul_legacy", "matmul", "add", "sub", "mul", "div_op",
                "sum_along_dim", "softmax_last_dim", "layer_norm"
            };
            SiliconContext.registerAll(device, tensorOpsModule, tensorOpsKernels);

            String[] scalarKernels = {
                "mask", "add_scalar", "sub_scalar", "mul_scalar",
                "div_scalar", "pow_scalar", "sqrt_op"
            };
            SiliconContext.registerAll(device, elementaryOpsModule, scalarKernels);

            for (Activations activation : Activations.values()) {
                Activation function = activation.function();
                String prefix = function.getKernelPrefix();

                SiliconContext.register(device, prefix + "_forward", activationsModule);
                SiliconContext.register(device, prefix + "_backward", activationsModule);
            }

            String[] gradientClipKernels = {
                "hard_clip", "l2_clip",
                "l2_norm_squared_reduce", "l2_norm_squared_final",
                "l2_norm_squared_single_block", "l2_norm_sqrt", "compute_l2_clip_scale"
            };
            SiliconContext.registerAll(device, gradientClipModule, gradientClipKernels);

            String[] flashAttentionKernels = {
                "flash_attention_forward", "flash_attention_forward_with_lse",
                "flash_attention_backward", "flash_attention_backward_dq",
                "flash_attention_forward_tiled", "flash_attention_backward_tiled"
            };
            SiliconContext.registerAll(device, flashAttentionModule, flashAttentionKernels);

            String[] fftKernels = {
                "fft1d", "bit_reverse_permute", "fft_butterfly_stage", "fft_normalize",
                "fft2d_rows", "fft2d_transpose", "fft2d_copy"
            };
            SiliconContext.registerAll(device, fftModule, fftKernels);

            String[] convolutionKernels = {
                "convolve1d_direct", "convolve2d_direct", "convolve2d_fft_extract",
                "convolve1d_fft_prepare", "convolve1d_fft_multiply", "convolve1d_fft_extract",
                "convolve2d_fft_prepare_input", "convolve2d_fft_prepare_kernel", "convolve2d_fft_multiply"
            };
            SiliconContext.registerAll(device, convolutionModule, convolutionKernels);

            String[] complexKernels = {
                "complex_pointwise_mul", "complex_pointwise_add"
            };
            SiliconContext.registerAll(device, complexOpsModule, complexKernels);

        } catch (Throwable e) {
            throw new RuntimeException("Failed to initialize GPU kernels", e);
        }
    }

    private int roundUp(int globalSize) {
        int r = globalSize % TILE_SIZE;
        if (r == 0) return globalSize;
        return globalSize + TILE_SIZE - r;
    }

    private Tensor launchScalarKernel(String kernelName, float value) {
        try (SiliconContext.QueueHandle qh = SiliconContext.getOrCreateQueue(device)) {
            int workSize = Math.max(1, size);

            SiliconKernel.create(device, kernelName)
                .addBuffer(dataBuffer)
                .addFloat(value)
                .addInt(size)
                .launch(qh.queue(), workSize);

            qh.queue().awaitCompletion();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to launch scalar kernel: " + kernelName, e);
        }
        return this;
    }

    private Tensor launchElementaryKernel(String kernelName, Tensor other) {
        if (!(other instanceof SiliconGpuTensor)) {
            other = other.to(device);
        }

        SiliconGpuTensor B = (SiliconGpuTensor) other;

        int broadcastDim = (Arrays.equals(shape, B.shape)) ? -1 : shape[1];
        int batch = (broadcastDim == -1) ? 0 : shape[0];

        try (SiliconContext.QueueHandle qh = SiliconContext.getOrCreateQueue(device)) {
            int workSize = Math.max(1, size);

            SiliconKernel.create(device, kernelName)
                .addBuffer(dataBuffer)
                .addBuffer(B.dataBuffer)
                .addInt(size)
                .addInt(broadcastDim)
                .addInt(batch)
                .launch(qh.queue(), workSize);

            qh.queue().awaitCompletion();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to launch elementary kernel: " + kernelName, e);
        }
        return this;
    }

    @Override
    public Tensor clone() {
        return new SiliconGpuTensor(device, shape, dataBuffer);
    }

    @Override
    public Tensor to(Object deviceObj) {
        if (deviceObj == null) {
            Tensor result = new CpuTensor(Shape.of(shape), data());
            result.setAutogradContext(autogradContext);
            return result;
        }

        if (deviceObj instanceof SiliconDevice targetDevice) {
            if (targetDevice.equals(device)) {
                return this;
            }
            // then we transfer it to a different device
            return new SiliconGpuTensor(targetDevice, shape, data());
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

        SiliconGpuTensor view = new SiliconGpuTensor(device, newShape, newStrides);

        // we share the data buffer
        dataBuffer.retain();
        view.dataBuffer.release();
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
        return launchElementaryKernel("div_op", other);
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
        // TODO
        return null;
    }

    @Override
    public Tensor sqrt() {
        try (SiliconContext.QueueHandle qh = SiliconContext.getOrCreateQueue(device)) {
            int workSize = Math.max(1, size);

            SiliconKernel.create(device, "sqrt_op")
                .addBuffer(dataBuffer)
                .addInt(size)
                .launch(qh.queue(), workSize);

            qh.queue().awaitCompletion();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to launch sqrt kernel", e);
        }
        return this;
    }

    @Override
    public Tensor activate(Activation activation) {
        return super.activate(activation);
    }

    @Override
    public Tensor matmul(Tensor other) {
        if (!(other instanceof SiliconGpuTensor B)) {
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

        SiliconGpuTensor result = new SiliconGpuTensor(device, outShape);

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

        SiliconBuffer memoryA = device.createBuffer(offsetsA);
        SiliconBuffer memoryB = device.createBuffer(offsetsB);
        SiliconBuffer memoryC = device.createBuffer(offsetsC);

        try (SiliconContext.QueueHandle qh = SiliconContext.getOrCreateQueue(device)) {
            ComputeSize globalSize = new ComputeSize(roundUp(M), roundUp(P), Math.max(1, batchCount));
            ComputeSize localSize = new ComputeSize(TILE_SIZE, TILE_SIZE, 1);

            SiliconKernel.create(device, "matmul_batched")
                .addBuffer(dataBuffer)
                .addBuffer(B.dataBuffer)
                .addBuffer(result.dataBuffer)
                .addBuffer(memoryA)
                .addBuffer(memoryB)
                .addBuffer(memoryC)
                .addInt(M)
                .addInt(K)
                .addInt(P)
                .addInt(batchCount)
                .addInt(transposed ? 1 : 0)
                .addInt(other.transposed() ? 1 : 0)
                .launch(qh.queue(), globalSize, localSize);

            qh.queue().awaitCompletion();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to launch matmul kernel", e);
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

        SiliconGpuTensor result = new SiliconGpuTensor(device, newShape);
        result.setAutogradContext(autogradContext);

        try (SiliconContext.QueueHandle qh = SiliconContext.getOrCreateQueue(device)) {
            int gx = Math.max(1, outerSize);
            int gy = Math.max(1, innerSize);

            SiliconKernel.create(device, "sum_along_dim")
                .addBuffer(dataBuffer)
                .addBuffer(result.dataBuffer)
                .addInt(outerSize)
                .addInt(reducedSize)
                .addInt(innerSize)
                .launch(qh.queue(), gx, gy);

            qh.queue().awaitCompletion();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to launch sum_along_dim kernel", e);
        }

        return result;
    }

    @Override
    public Tensor reshape(int... newShape) {
        int newSize = Tensors.computeSize(newShape);

        if (newSize != size) {
            throw new IllegalArgumentException(
                "The total new dimension (" + newSize + ") does not match the current dimension (" + size + ")"
            );
        }

        return new SiliconGpuTensor(this, newShape);
    }

    @Override
    public Tensor concat(Tensor other) {
        if (!(other instanceof SiliconGpuTensor)) {
            other = other.to(device);
        }

        SiliconGpuTensor B = (SiliconGpuTensor) other;

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

        int outerSize =
                stream(shape, 0, rank - 1)
                .reduce(1, (a, b) -> a * b);

        SiliconGpuTensor result = new SiliconGpuTensor(device, newShape);

        try (SiliconContext.QueueHandle qh = SiliconContext.getOrCreateQueue(device)) {
            int workSize = Math.max(1, outerSize * concatLast);

            SiliconKernel.create(device, "concat_last_dim")
                .addBuffer(this.dataBuffer)
                .addBuffer(B.dataBuffer)
                .addBuffer(result.dataBuffer)
                .addInt(outerSize)
                .addInt(lastA)
                .addInt(lastB)
                .addInt(concatLast)
                .launch(qh.queue(), workSize);

            qh.queue().awaitCompletion();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to launch concat kernel", e);
        }

        return result;
    }

    @Override
    public Tensor concat(Tensor other, int dimension) {
        if (!(other instanceof SiliconGpuTensor)) {
            other = other.to(device);
        }

        SiliconGpuTensor B = (SiliconGpuTensor) other;

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

        int blockSize =
                stream(shape, dimension + 1, rank)
                .reduce(1, (a, b) -> a * b);

        int numBlocks =
                stream(shape, 0, dimension)
                .reduce(1, (a, b) -> a * b);

        int thisDim = shape[dimension];
        int otherDim = B.shape[dimension];

        SiliconGpuTensor result = new SiliconGpuTensor(device, newShape);

        int totalA = numBlocks * thisDim * blockSize;
        int totalB = numBlocks * otherDim * blockSize;

        try (SiliconContext.QueueHandle qh = SiliconContext.getOrCreateQueue(device)) {
            SiliconKernel.create(device, "concat_copy_a")
                .addBuffer(this.dataBuffer)
                .addBuffer(result.dataBuffer)
                .addInt(numBlocks)
                .addInt(thisDim)
                .addInt(otherDim)
                .addInt(blockSize)
                .launch(qh.queue(), Math.max(1, totalA));

            SiliconKernel.create(device, "concat_copy_b")
                .addBuffer(B.dataBuffer)
                .addBuffer(result.dataBuffer)
                .addInt(numBlocks)
                .addInt(thisDim)
                .addInt(otherDim)
                .addInt(blockSize)
                .launch(qh.queue(), Math.max(1, totalB));

            qh.queue().awaitCompletion();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to launch concat kernel", e);
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

        SiliconGpuTensor result = new SiliconGpuTensor(device, newShape);

        int[] starts = new int[ranges.length];
        int[] steps = new int[ranges.length];

        for (int i = 0; i < ranges.length; i++) {
            Range range = ranges[i];
            starts[i] = range == null ? 0 : range.start();
            steps[i] = range == null ? 1 : range.step();
        }

        SiliconBuffer memoryShape = device.createBuffer(newShape);
        SiliconBuffer memoryStart = device.createBuffer(starts);
        SiliconBuffer memoryStep = device.createBuffer(steps);

        try (SiliconContext.QueueHandle qh = SiliconContext.getOrCreateQueue(device)) {
            int workSize = Math.max(1, result.elements());

            SiliconKernel.create(device, "slice")
                .addBuffer(this.dataBuffer)
                .addBuffer(result.dataBuffer)
                .addBuffer(this.stridesBuffer)
                .addBuffer(result.stridesBuffer)
                .addBuffer(memoryShape)
                .addBuffer(memoryStart)
                .addBuffer(memoryStep)
                .addInt(rank())
                .launch(qh.queue(), workSize);

            qh.queue().awaitCompletion();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to launch slice kernel", e);
        }

        memoryShape.release();
        memoryStart.release();
        memoryStep.release();

        return result;
    }

    @Override
    public Tensor layerNorm(double epsilon) {
        SiliconGpuTensor result = new SiliconGpuTensor(device, shape);
        result.setAutogradContext(autogradContext);

        int rank = shape.length;
        int featuresSize = shape[rank - 1];
        int batchSize = 1;

        for (int i = 0; i < rank - 1; i++) batchSize *= shape[i];

        try (SiliconContext.QueueHandle qh = SiliconContext.getOrCreateQueue(device)) {
            int workSize = Math.max(1, batchSize);

            SiliconKernel.create(device, "layer_norm")
                .addBuffer(this.dataBuffer)
                .addBuffer(result.dataBuffer)
                .addInt(featuresSize)
                .addInt(batchSize)
                .addFloat((float) epsilon)
                .launch(qh.queue(), workSize);

            qh.queue().awaitCompletion();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to launch layer_norm kernel", e);
        }

        return result;
    }

    @Override
    public float[] data() {
        float[] buffer = new float[size];
        dataBuffer.get(buffer);
        return buffer;
    }

    @Override
    public Tensor set(float value, int... indices) {
        // TODO
        return null;
    }

    @Override
    public int elements() {
        return size;
    }

    @Override
    public Tensor softmax(double temperature) {
        SiliconGpuTensor result = new SiliconGpuTensor(device, shape);

        int lastDim = shape[shape.length - 1];
        int rows = size / lastDim;

        try (SiliconContext.QueueHandle qh = SiliconContext.getOrCreateQueue(device)) {
            int workSize = Math.max(1, rows);
            SiliconKernel.create(device, "softmax_last_dim")
                .addBuffer(dataBuffer)
                .addBuffer(result.dataBuffer)
                .addInt(lastDim)
                .addFloat((float) temperature)
                .launch(qh.queue(), workSize);

            qh.queue().awaitCompletion();
        } catch (Throwable e) {
            throw new RuntimeException("Failed to launch softmax kernel", e);
        }

        return result;
    }
}

