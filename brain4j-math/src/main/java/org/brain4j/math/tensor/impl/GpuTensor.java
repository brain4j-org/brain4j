package org.brain4j.math.tensor.impl;

import org.brain4j.math.Tensors;
import org.brain4j.math.activation.Activation;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.gpu.GpuContext;
import org.brain4j.math.gpu.device.Device;
import org.brain4j.math.gpu.kernel.KernelFactory;
import org.brain4j.math.tensor.Shape;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.commons.Range;
import org.brain4j.math.tensor.TensorKey;
import org.brain4j.math.tensor.Usage;
import org.silicon.api.device.ComputeBuffer;
import org.silicon.api.backend.BackendType;
import org.silicon.api.function.ComputeModule;
import org.silicon.api.kernel.ComputeSize;
import org.silicon.api.slang.SlangCompiler;

import java.util.Arrays;

import static java.util.Arrays.stream;

public class GpuTensor extends BaseTensor {

    private static final int TILE_SIZE = 16;

    private final Device device;
    private final ComputeBuffer stridesBuffer;
    private final ComputeBuffer dataBuffer;
    private final boolean persistent;
    private final int size;

    public GpuTensor(Device device, int[] shape, float... data) {
        this(device, false, shape, data);
    }

    public GpuTensor(Device device, int[] shape, int[] strides, float... data) {
        this(device, shape, strides);
        if (data.length != 0) {
            getDataBuffer().write(data);
        }
    }

    public GpuTensor(Device device, boolean persistent, int[] shape, float... data) {
        this.device = device;
        this.persistent = persistent;
        this.size = Tensors.computeSize(shape);
        this.shape = shape;
        this.strides = Tensors.computeStrides(shape);

        TensorKey stridesKey = new TensorKey(Usage.STRIDES, shape);
        TensorKey dataKey = new TensorKey(Usage.DATA, shape);

        if (persistent) {
            // Persistent tensors (e.g. model params, optimizer state) must not be returned to
            // the batch-level pool, otherwise they can be reused while still logically alive
            this.stridesBuffer = device.createBuffer(strides);
            this.dataBuffer = device.createBuffer(data);
        } else {
            this.stridesBuffer = device.acquire(stridesKey, () -> device.createBuffer(strides));
            this.dataBuffer = device.acquire(dataKey, () -> {
                float[] newData = data.length != 0 ? data : new float[size];
                return device.createBuffer(newData);
            });
            if (data.length != 0) this.dataBuffer.write(data);
        }
    }

    public GpuTensor(Device device, int[] shape, ComputeBuffer otherBuffer) {
        this.device = device;
        this.persistent = false;
        this.size = Tensors.computeSize(shape);
        this.shape = shape;
        this.strides = Tensors.computeStrides(shape);

        TensorKey stridesKey = new TensorKey(Usage.STRIDES, shape);
        TensorKey dataKey = new TensorKey(Usage.DATA, shape);

        this.stridesBuffer = device.acquire(stridesKey, () -> device.createBuffer(strides));
        this.dataBuffer = device.acquire(dataKey, otherBuffer::copy);
        otherBuffer.copyInto(dataBuffer);
    }

    public GpuTensor(Device device, int[] shape, int[] strides) {
        this.device = device;
        this.persistent = false;
        this.size = Tensors.computeSize(shape);
        this.shape = shape;
        this.strides = strides;

        TensorKey stridesKey = new TensorKey(Usage.STRIDES, shape);
        TensorKey dataKey = new TensorKey(Usage.DATA, shape);

        long dataSize = (long) Float.BYTES * this.size;

        this.stridesBuffer = device.acquire(stridesKey, () -> device.createBuffer(strides));
        this.dataBuffer = device.acquire(dataKey, () -> device.createBuffer(dataSize));
    }

    public GpuTensor(GpuTensor reference, int[] newShape) {
        this.device = reference.device;
        this.persistent = false;
        this.size = Tensors.computeSize(newShape);
        this.shape = newShape;
        this.strides = Tensors.computeStrides(newShape);

        TensorKey stridesKey = new TensorKey(Usage.STRIDES, newShape);

        this.stridesBuffer = device.acquire(stridesKey, () -> device.createBuffer(strides));
        this.dataBuffer = reference.dataBuffer;
    }

    private GpuTensor(GpuTensor reference, int[] newShape, int[] newStrides) {
        this.device = reference.device;
        this.persistent = false;
        this.size = Tensors.computeSize(newShape);
        this.shape = newShape;
        this.strides = newStrides;

        TensorKey stridesKey = new TensorKey(Usage.STRIDES, newShape);

        this.stridesBuffer = device.acquire(stridesKey, () -> device.createBuffer(newStrides));
        this.dataBuffer = reference.dataBuffer;
    }

    public static GpuTensor persistent(Tensor first, Device device) {
        return new GpuTensor(device, true, first.shape(), first.data());
    }

    public boolean isPersistent() {
        return persistent;
    }

    public Device getDevice() {
        return device;
    }

    public Device device() {
        return device;
    }

    public ComputeBuffer getDataBuffer() {
        return dataBuffer;
    }

    public ComputeBuffer getStridesBuffer() {
        return stridesBuffer;
    }

    public int size() {
        return size;
    }

    public static void initKernels(Device device) {
        try {
            SlangCompiler compiler = new SlangCompiler(device.context());

            // TODO: hope that Slang devs wake the fuck up and fix this
            if (device.context().backendType() == BackendType.METAL) {
                // Metal: per-kernel compilation avoids global buffer(>30) overflow
                // Each file contains exactly one entry point -> bindings start at 0
                String[] activationKernels = {
                    "activation_forward", "activation_backward",
                    "elu_forward", "elu_backward", "gelu_forward", "gelu_backward",
                    "leakyrelu_forward", "leakyrelu_backward", "linear_forward", "linear_backward",
                    "mish_forward", "mish_backward", "relu_forward", "relu_backward",
                    "sigmoid_forward", "sigmoid_backward", "softmax_forward", "softmax_backward",
                    "softplus_forward", "softplus_backward", "swish_forward", "swish_backward",
                    "tanh_forward", "tanh_backward"
                };
                for (String k : activationKernels) {
                    ComputeModule m = compiler.compileFromResource("shaders/activations/" + k + ".slang");
                    GpuContext.storeModule(device, "activations_" + k, m);
                    GpuContext.register(device, k, m);
                }

                String[] tensorOpsKernels = {
                    "matmul", "matmul_batched", "matmul_legacy",
                    "slice", "slice_backward_scatter", "layer_norm", "broadcast_to",
                    "add_op", "sub_op", "mul_op", "div_op", "pow_op",
                    "add", "sub", "mul", "div",
                    "sum_along_dim", "softmax_last_dim",
                    "concat_last_dim", "concat_copy_a", "concat_copy_b", "copy_strided"
                };
                for (String k : tensorOpsKernels) {
                    ComputeModule m = compiler.compileFromResource("shaders/tensor_ops/" + k + ".slang");
                    GpuContext.storeModule(device, "tensor_ops_" + k, m);
                    GpuContext.register(device, k, m);
                }

                String[] scalarKernels = {
                    "mask", "add_scalar", "sub_scalar", "mul_scalar",
                    "div_scalar", "pow_scalar", "sqrt_op"
                };
                for (String k : scalarKernels) {
                    ComputeModule m = compiler.compileFromResource("shaders/elementary_ops/" + k + ".slang");
                    GpuContext.storeModule(device, "elementary_ops_" + k, m);
                    GpuContext.register(device, k, m);
                }

                String[] gradientClipKernels = {
                    "hard_clip", "l2_clip", "l2_clip_from_scale",
                    "l2_norm_squared_reduce", "l2_sum_reduce", "l2_norm_squared_final",
                    "l2_norm_squared_single_block", "l2_norm_sqrt", "compute_l2_clip_scale"
                };
                for (String k : gradientClipKernels) {
                    ComputeModule m = compiler.compileFromResource("shaders/gradient_clippers/" + k + ".slang");
                    GpuContext.storeModule(device, "gradient_clippers_" + k, m);
                    GpuContext.register(device, k, m);
                }

                String[] convolutionKernels = {
                    "conv2d_nchw", "conv2d_nchw_shared", "conv2d_backward_input_nchw", "conv2d_backward_filter_nchw",
                    "convolve1d_direct", "convolve1d_fft", "convolve2d_direct", "convolve2d_fft_extract"
                };
                for (String k : convolutionKernels) {
                    ComputeModule m = compiler.compileFromResource("shaders/convolution/" + k + ".slang");
                    GpuContext.storeModule(device, "convolution_" + k, m);
                    GpuContext.register(device, k, m);
                }

                ComputeModule complexM = compiler.compileFromResource("shaders/complex_ops/complex_pointwise_mul.slang");
                GpuContext.storeModule(device, "complex_ops", complexM);
                GpuContext.register(device, "complex_pointwise_mul", complexM);

                String[] flashAttentionKernels = {
                    "flash_attention_forward", "flash_attention_forward_with_lse",
                    "flash_attention_backward", "flash_attention_backward_dq",
                    "flash_attention_forward_tiled", "flash_attention_backward_tiled"
                };
                for (String k : flashAttentionKernels) {
                    ComputeModule m = compiler.compileFromResource("shaders/flash_attention/" + k + ".slang");
                    GpuContext.storeModule(device, "flash_attention_" + k, m);
                    GpuContext.register(device, k, m);
                }

                String[] poolingKernels = { "max_pool_forward", "max_pool_backward" };
                for (String k : poolingKernels) {
                    ComputeModule m = compiler.compileFromResource("shaders/pooling/" + k + ".slang");
                    GpuContext.storeModule(device, "pooling_" + k, m);
                    GpuContext.register(device, k, m);
                }
                return;
            }

            // CUDA / OpenCL: single-module compilation (fast, no buffer limit)
            ComputeModule tensorOpsModule = compiler.compileFromResource("shaders/tensor_ops.slang");
            ComputeModule elementaryOpsModule = compiler.compileFromResource("shaders/elementary_ops.slang");
            ComputeModule activationsModule = compiler.compileFromResource("shaders/activations.slang");
            ComputeModule gradientClipModule = compiler.compileFromResource("shaders/gradient_clippers.slang");
            ComputeModule convolutionModule = compiler.compileFromResource("shaders/convolution.slang");
            ComputeModule complexOpsModule = compiler.compileFromResource("shaders/complex_ops.slang");
            ComputeModule flashAttentionModule = compiler.compileFromResource("shaders/flash_attention.slang");
            ComputeModule poolingModule = compiler.compileFromResource("shaders/pooling.slang");

            GpuContext.storeModule(device, "tensor_ops", tensorOpsModule);
            GpuContext.storeModule(device, "elementary_ops", elementaryOpsModule);
            GpuContext.storeModule(device, "activations", activationsModule);
            GpuContext.storeModule(device, "gradient_clippers", gradientClipModule);
            GpuContext.storeModule(device, "convolution", convolutionModule);
            GpuContext.storeModule(device, "complex_ops", complexOpsModule);
            GpuContext.storeModule(device, "flash_attention", flashAttentionModule);
            GpuContext.storeModule(device, "pooling", poolingModule);

            GpuContext.register(device, "activation_forward", activationsModule);
            GpuContext.register(device, "activation_backward", activationsModule);

            String[] activationKernels = {
                "elu_forward", "elu_backward", "gelu_forward", "gelu_backward",
                "leakyrelu_forward", "leakyrelu_backward", "linear_forward", "linear_backward",
                "mish_forward", "mish_backward", "relu_forward", "relu_backward",
                "sigmoid_forward", "sigmoid_backward", "softmax_forward", "softmax_backward",
                "softplus_forward", "softplus_backward", "swish_forward", "swish_backward",
                "tanh_forward", "tanh_backward"
            };
            GpuContext.registerAll(device, activationsModule, activationKernels);

            String[] tensorOpsKernels = {
                "matmul", "matmul_batched", "matmul_legacy",
                "slice", "slice_backward_scatter", "layer_norm", "broadcast_to",
                "add_op", "sub_op", "mul_op", "div_op", "pow_op",
                "add", "sub", "mul", "div",
                "sum_along_dim", "softmax_last_dim",
                "concat_last_dim", "concat_copy_a", "concat_copy_b", "copy_strided"
            };
            GpuContext.registerAll(device, tensorOpsModule, tensorOpsKernels);

            String[] scalarKernels = {
                "mask", "add_scalar", "sub_scalar", "mul_scalar",
                "div_scalar", "pow_scalar", "sqrt_op"
            };
            GpuContext.registerAll(device, elementaryOpsModule, scalarKernels);

            String[] gradientClipKernels = {
                "hard_clip", "l2_clip", "l2_clip_from_scale",
                "l2_norm_squared_reduce", "l2_sum_reduce", "l2_norm_squared_final",
                "l2_norm_squared_single_block", "l2_norm_sqrt", "compute_l2_clip_scale"
            };
            GpuContext.registerAll(device, gradientClipModule, gradientClipKernels);

            String[] convolutionKernels = {
                "conv2d_nchw", "conv2d_nchw_shared", "conv2d_backward_input_nchw", "conv2d_backward_filter_nchw",
                "convolve1d_direct", "convolve1d_fft", "convolve2d_direct", "convolve2d_fft_extract"
            };
            GpuContext.registerAll(device, convolutionModule, convolutionKernels);

            GpuContext.register(device, "complex_pointwise_mul", complexOpsModule);

            String[] flashAttentionKernels = {
                "flash_attention_forward", "flash_attention_forward_with_lse",
                "flash_attention_backward", "flash_attention_backward_dq",
                "flash_attention_forward_tiled", "flash_attention_backward_tiled"
            };
            GpuContext.registerAll(device, flashAttentionModule, flashAttentionKernels);

            String[] poolingKernels = { "max_pool_forward", "max_pool_backward" };
            GpuContext.registerAll(device, poolingModule, poolingKernels);
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
        try (GpuContext.QueueHandle qh = GpuContext.getOrCreateQueue(device)) {
            int workSize = Math.max(1, size);

            KernelFactory.create(device, kernelName)
                .buffer(dataBuffer)
                .floatVal(value)
                .intVal(size)
                .launch(qh.queue(), workSize);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to launch scalar kernel: " + kernelName, e);
        }
        return this;
    }

    private Tensor launchElementaryKernel(String kernelName, Tensor other) {
        if (!(other instanceof GpuTensor B)) {
            return launchElementaryKernel(kernelName, other.to(device));
        }

        int rankA = shape.length;
        int rankB = B.shape.length;

        for (int d = 0; d < rankA; d++) {
            int db = d - (rankA - rankB);
            int bDim = db >= 0 ? B.shape[db] : 1;

            if (shape[d] != bDim && bDim != 1) {
                throw new IllegalArgumentException(
                    "Incompatible shapes for broadcasting: " + Arrays.toString(shape) + " vs " + Arrays.toString(B.shape)
                );
            }
        }

        int[] aFlatStrides = Tensors.computeStrides(shape);
        int[] bEffStrides = new int[rankA];
        int dimOffset = rankA - rankB;

        for (int d = 0; d < rankA; d++) {
            int db = d - dimOffset;
            if (db < 0) {
                bEffStrides[d] = 0;
            } else if (B.shape[db] == 1) {
                bEffStrides[d] = 0;
            } else {
                bEffStrides[d] = B.strides[db];
            }
        }

        int sameShape = Arrays.equals(shape, B.shape) ? 1 : 0;

        TensorKey aShapeKey = new TensorKey(Usage.OTHER, shape);
        TensorKey aFlatStridesKey = new TensorKey(Usage.OTHER, aFlatStrides);
        TensorKey bEffStridesKey = new TensorKey(Usage.OTHER, bEffStrides);

        ComputeBuffer aShapeBuffer = device.acquire(aShapeKey, () -> device.createBuffer(shape));
        ComputeBuffer aFlatStridesBuffer = device.acquire(aFlatStridesKey, () -> device.createBuffer(aFlatStrides));
        ComputeBuffer bEffStridesBuffer = device.acquire(bEffStridesKey, () -> device.createBuffer(bEffStrides));

        try (GpuContext.QueueHandle qh = GpuContext.getOrCreateQueue(device)) {
            int workSize = Math.max(1, size);

            KernelFactory.create(device, kernelName)
                .buffer(dataBuffer)
                .buffer(B.dataBuffer)
                .intVal(size)
                .buffer(aShapeBuffer)
                .buffer(aFlatStridesBuffer)
                .buffer(bEffStridesBuffer)
                .intVal(rankA)
                .intVal(sameShape)
                .launch(qh.queue(), workSize);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to launch elementary kernel: " + kernelName, e);
        }
        return this;
    }

    @Override
    public Tensor to(Object deviceObj) {
        if (deviceObj == null) {
            Tensor result = new CpuTensor(Shape.of(shape), contiguousData());
            result.setAutogradContext(autogradContext);
            return result;
        }

        if (deviceObj instanceof Device targetDevice) {
            if (targetDevice == device) {
                return this;
            }

            GpuTensor result = new GpuTensor(targetDevice, shape, contiguousData());
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

        GpuTensor view = new GpuTensor(this, newShape, newStrides);
        view.transposed = !transposed;

        return view;
    }

    @Override
    public Tensor add(Tensor other) {
        return launchElementaryKernel("add_op", other);
    }

    @Override
    public Tensor add(double value) {
        return launchScalarKernel("add_scalar", (float) value);
    }

    @Override
    public Tensor sub(Tensor other) {
        return launchElementaryKernel("sub_op", other);
    }

    @Override
    public Tensor sub(double value) {
        return launchScalarKernel("sub_scalar", (float) value);
    }

    @Override
    public Tensor mul(Tensor other) {
        return launchElementaryKernel("mul_op", other);
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
    public Tensor pow(Tensor other) {
        return launchElementaryKernel("pow_op", other);
    }

    @Override
    public Tensor pow(double value) {
        return launchScalarKernel("pow_scalar", (float) value);
    }

    @Override
    public Tensor sqrt() {
        try (GpuContext.QueueHandle qh = GpuContext.getOrCreateQueue(device)) {
            int workSize = Math.max(1, size);

            KernelFactory.create(device, "sqrt_op")
                .buffer(dataBuffer)
                .intVal(size)
                .launch(qh.queue(), workSize);
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
        if (!(other instanceof GpuTensor B)) {
            return matmul(other.to(device));
        }

        // Stride-aware matmul: the kernel resolves the real layout of A and B,
        // so views (transposed, sliced) are read directly from their buffers —
        // no materialization, no copies, identical semantics on every backend.
        int[] shapeA = shape;
        int[] shapeB = B.shape;

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

        try (GpuContext.QueueHandle qh = GpuContext.getOrCreateQueue(device)) {
            ComputeSize globalSize = new ComputeSize(roundUp(M), roundUp(P), Math.max(1, batchCount));
            ComputeSize localSize = new ComputeSize(TILE_SIZE, TILE_SIZE, 1);

            if (batchCount == 1 && maxBatchRank == 0) {
                TensorKey stridesAKey = new TensorKey(Usage.OTHER, strides);
                TensorKey stridesBKey = new TensorKey(Usage.OTHER, B.strides);

                ComputeBuffer stridesA = device.acquire(stridesAKey, () -> device.createBuffer(strides));
                ComputeBuffer stridesB = device.acquire(stridesBKey, () -> device.createBuffer(B.strides));

                KernelFactory.create(device, "matmul")
                    .buffer(dataBuffer)
                    .buffer(B.dataBuffer)
                    .buffer(result.dataBuffer)
                    .buffer(stridesA)
                    .buffer(stridesB)
                    .intVal(M)
                    .intVal(K)
                    .intVal(P)
                    .launch(qh.queue(), globalSize, localSize);
            } else {
                TensorKey memoryAKey = new TensorKey(Usage.OTHER, offsetsA);
                TensorKey memoryBKey = new TensorKey(Usage.OTHER, offsetsB);
                TensorKey memoryCKey = new TensorKey(Usage.OTHER, offsetsC);

                ComputeBuffer memoryA = device.acquire(memoryAKey, () -> device.createBuffer(offsetsA));
                ComputeBuffer memoryB = device.acquire(memoryBKey, () -> device.createBuffer(offsetsB));
                ComputeBuffer memoryC = device.acquire(memoryCKey, () -> device.createBuffer(offsetsC));

                KernelFactory.create(device, "matmul_batched")
                    .buffer(dataBuffer)
                    .buffer(B.dataBuffer)
                    .buffer(result.dataBuffer)
                    .buffer(memoryA)
                    .buffer(memoryB)
                    .buffer(memoryC)
                    .intVal(M)
                    .intVal(K)
                    .intVal(P)
                    .intVal(batchCount)
//                    .intVal(0)
//                    .intVal(0)
                    .intVal(transposed ? 1 : 0)
                    .intVal(other.transposed() ? 1 : 0)
                    .launch(qh.queue(), globalSize, localSize);
            }
        } catch (Throwable e) {
            throw new RuntimeException("Failed to launch matmul kernel", e);
        }

        return result;
    }

    @Override
    public Tensor sum(int dim, boolean keepDim) {
        dim = Commons.mod(dim, shape.length);

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

        try (GpuContext.QueueHandle qh = GpuContext.getOrCreateQueue(device)) {
            int gx = Math.max(1, outerSize);
            int gy = Math.max(1, innerSize);

            KernelFactory.create(device, "sum_along_dim")
                .buffer(dataBuffer)
                .buffer(result.dataBuffer)
                .intVal(outerSize)
                .intVal(reducedSize)
                .intVal(innerSize)
                .launch(qh.queue(), gx, gy);
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

        int outerSize =
                stream(shape, 0, rank - 1)
                .reduce(1, (a, b) -> a * b);

        GpuTensor result = new GpuTensor(device, newShape);

        try (GpuContext.QueueHandle qh = GpuContext.getOrCreateQueue(device)) {
            int workSize = Math.max(1, outerSize * concatLast);

            KernelFactory.create(device, "concat_last_dim")
                .buffer(this.dataBuffer)
                .buffer(B.dataBuffer)
                .buffer(result.dataBuffer)
                .intVal(outerSize)
                .intVal(lastA)
                .intVal(lastB)
                .intVal(concatLast)
                .launch(qh.queue(), workSize);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to launch concat kernel", e);
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

        int blockSize =
                stream(shape, dimension + 1, rank)
                .reduce(1, (a, b) -> a * b);

        int numBlocks =
                stream(shape, 0, dimension)
                .reduce(1, (a, b) -> a * b);

        int thisDim = shape[dimension];
        int otherDim = B.shape[dimension];

        GpuTensor result = new GpuTensor(device, newShape);

        int totalA = numBlocks * thisDim * blockSize;
        int totalB = numBlocks * otherDim * blockSize;

        try (GpuContext.QueueHandle qh = GpuContext.getOrCreateQueue(device)) {
            KernelFactory.create(device, "concat_copy_a")
                .buffer(this.dataBuffer)
                .buffer(result.dataBuffer)
                .intVal(numBlocks)
                .intVal(thisDim)
                .intVal(otherDim)
                .intVal(blockSize)
                .launch(qh.queue(), Math.max(1, totalA));

            KernelFactory.create(device, "concat_copy_b")
                .buffer(B.dataBuffer)
                .buffer(result.dataBuffer)
                .intVal(numBlocks)
                .intVal(thisDim)
                .intVal(otherDim)
                .intVal(blockSize)
                .launch(qh.queue(), Math.max(1, totalB));
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

        GpuTensor result = new GpuTensor(device, newShape);

        int[] starts = new int[shape.length];
        int[] steps = new int[shape.length];

        for (int i = 0; i < shape.length; i++) {
            Range range = i < ranges.length ? ranges[i] : null;
            starts[i] = range == null ? 0 : range.start(shape[i]);
            steps[i] = range == null ? 1 : range.step();
        }

        TensorKey shapeKey = new TensorKey(Usage.OTHER, newShape);
        TensorKey startKey = new TensorKey(Usage.OTHER, starts);
        TensorKey stepKey = new TensorKey(Usage.OTHER, steps);

        ComputeBuffer memoryShape = device.acquire(shapeKey, () -> device.createBuffer(newShape));
        ComputeBuffer memoryStart = device.acquire(startKey, () -> device.createBuffer(starts));
        ComputeBuffer memoryStep = device.acquire(stepKey, () -> device.createBuffer(steps));

        try (GpuContext.QueueHandle qh = GpuContext.getOrCreateQueue(device)) {
            int workSize = Math.max(1, result.elements());

            KernelFactory.create(device, "slice")
                .buffer(this.dataBuffer)
                .buffer(result.dataBuffer)
                .buffer(this.stridesBuffer)
                .buffer(result.stridesBuffer)
                .buffer(memoryShape)
                .buffer(memoryStart)
                .buffer(memoryStep)
                .intVal(rank())
                .launch(qh.queue(), workSize);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to launch slice kernel", e);
        }

        return result;
    }

    @Override
    public Tensor layerNorm(double epsilon) {
        int rank = shape.length;
        int featuresSize = shape[rank - 1];
        int batchSize = 1;

        for (int i = 0; i < rank - 1; i++) batchSize *= shape[i];

        try (GpuContext.QueueHandle qh = GpuContext.getOrCreateQueue(device)) {
            int workSize = Math.max(1, batchSize);

            KernelFactory.create(device, "layer_norm")
                .buffer(this.dataBuffer)
                .buffer(this.dataBuffer)
                .intVal(featuresSize)
                .intVal(batchSize)
                .floatVal((float) epsilon)
                .launch(qh.queue(), workSize);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to launch layer_norm kernel", e);
        }

        return this;
    }

    @Override
    public float[] data() {
        device.queue().await();
        float[] buffer = new float[size];
        dataBuffer.get(buffer);
        return buffer;
    }

    @Override
    public Tensor set(float value, int... indices) {
        // TODO: maybe make a proper kernel for this?
        device.queue().await();

        int offset = linearIndex(indices);
        float[] buffer = new float[size];

        dataBuffer.get(buffer);
        buffer[offset] = value;
        dataBuffer.write(buffer);
        return this;
    }

    @Override
    public int elements() {
        return size;
    }

    @Override
    public Tensor softmax(double temperature) {
        GpuTensor result = new GpuTensor(device, shape);

        int rank = shape.length;
        int lastDim = shape[shape.length - 1];
        int rows = size / lastDim;
        int lastStride = strides[rank - 1];

        int[] rowOffsets = new int[rows];
        if (rows > 1) {
            int outerRank = rank - 1;
            int acc = 1;
            int[] outerStrides = new int[outerRank];

            for (int d = outerRank - 1; d >= 0; d--) {
                outerStrides[d] = acc;
                acc *= shape[d];
            }

            for (int row = 0; row < rows; row++) {
                int base = 0;
                for (int d = 0; d < outerRank; d++) {
                    int idx = (row / outerStrides[d]) % shape[d];
                    base += idx * strides[d];
                }
                rowOffsets[row] = base;
            }
        }

        TensorKey rowOffsetsKey = new TensorKey(Usage.OTHER, rowOffsets);
        ComputeBuffer rowOffsetsBuffer = device.acquire(rowOffsetsKey, () -> device.createBuffer(rowOffsets));

        try (GpuContext.QueueHandle qh = GpuContext.getOrCreateQueue(device)) {
            int workSize = Math.max(1, rows);
            KernelFactory.create(device, "softmax_last_dim")
                .buffer(dataBuffer)
                .buffer(result.dataBuffer)
                .buffer(rowOffsetsBuffer)
                .intVal(lastStride)
                .intVal(rows)
                .intVal(lastDim)
                .floatVal((float) temperature)
                .launch(qh.queue(), workSize);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to launch softmax kernel", e);
        }

        return result;
    }

    @Override
    public Tensor convolve(Tensor other) {
        return convolve(other, 1);
    }

    @Override
    public Tensor convolve(Tensor other, int stride) {
        if (stride <= 0) {
            throw Commons.illegalArgument("Stride must be > 0. Got: %s", stride);
        }

        if (!(other instanceof GpuTensor K)) {
            return convolve(other.to(device), stride);
        }

        Tensor input4 = this;
        Tensor kernel4 = K;

        while (input4.rank() < 4) input4 = input4.unsqueeze();
        while (kernel4.rank() < 4) kernel4 = kernel4.unsqueeze();

        GpuTensor A = (GpuTensor) input4;
        GpuTensor B = (GpuTensor) kernel4;

        int[] aShape = A.shape;
        int[] bShape = B.shape;

        int[] aStdStrides = Tensors.computeStrides(aShape);
        int[] bStdStrides = Tensors.computeStrides(bShape);
        boolean contiguousA = Arrays.equals(A.strides, aStdStrides);
        boolean contiguousB = Arrays.equals(B.strides, bStdStrides);

        if (!contiguousA || !contiguousB) {
            return Tensors.convolve(A.to(null), B.to(null), stride).to(device);
        }

        int batch = aShape[0];
        int inChannels = aShape[1];
        int inHeight = aShape[2];
        int inWidth = aShape[3];

        int numFilters = bShape[0];
        int kernelChannels = bShape[1];
        int kernelHeight = bShape[2];
        int kernelWidth = bShape[3];

        if (inChannels != kernelChannels) {
            throw Commons.illegalArgument(
                "Convolution channel mismatch: input channels %s, kernel channels %s", inChannels, kernelChannels
            );
        }

        int outHeight = (inHeight - kernelHeight) / stride + 1;
        int outWidth = (inWidth - kernelWidth) / stride + 1;

        if (outHeight <= 0 || outWidth <= 0) {
            throw Commons.illegalArgument(
                "Invalid convolution output shape: [%s, %s]", outHeight, outWidth
            );
        }

        GpuTensor out = new GpuTensor(device, new int[] { batch, numFilters, outHeight, outWidth });

        try (GpuContext.QueueHandle qh = GpuContext.getOrCreateQueue(device)) {
            String kernel = "conv2d_nchw_shared";
            int tile = 16;

            ComputeSize localSize = new ComputeSize(tile, tile, 1);
            ComputeSize globalSize;

            int groupsX = (outWidth + tile - 1) / tile;
            int groupsY = (outHeight + tile - 1) / tile;
            int groupsZ = batch * numFilters;

            globalSize = new ComputeSize(
                groupsX * tile,
                groupsY * tile,
                groupsZ
            );

            KernelFactory.create(device, kernel)
                .buffer(A.dataBuffer)
                .buffer(B.dataBuffer)
                .buffer(out.dataBuffer)
                .intVal(batch)
                .intVal(inChannels)
                .intVal(inHeight)
                .intVal(inWidth)
                .intVal(numFilters)
                .intVal(kernelHeight)
                .intVal(kernelWidth)
                .intVal(stride)
                .intVal(outHeight)
                .intVal(outWidth)
                .launch(qh.queue(), globalSize, localSize);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to launch convolution kernel", e);
        }

        return out;
    }

    @Override
    public Tensor broadcast(int[] targetShape) {
        if (Arrays.equals(shape, targetShape)) {
            return this;
        }

        int targetRank = targetShape.length;
        int srcRank = shape.length;

        if (targetRank < srcRank) {
            throw new IllegalArgumentException(
                "Cannot broadcast: target rank " + targetRank + " is smaller than source rank " + srcRank
            );
        }

        int[] alignedSrcShape = new int[targetRank];
        int[] alignedSrcStrides = new int[targetRank];
        int[] targetFlatStrides = Tensors.computeStrides(targetShape);

        int pad = targetRank - srcRank;
        for (int i = 0; i < pad; i++) {
            alignedSrcShape[i] = 1;
            alignedSrcStrides[i] = 0;
        }

        for (int i = 0; i < srcRank; i++) {
            alignedSrcShape[pad + i] = shape[i];
            alignedSrcStrides[pad + i] = strides[i];
        }

        for (int d = 0; d < targetRank; d++) {
            if (alignedSrcShape[d] > targetShape[d]) {
                throw new IllegalArgumentException(
                    "Cannot broadcast: source dimension " + alignedSrcShape[d] +
                    " > target dimension " + targetShape[d] + " at axis " + d
                );
            }
        }

        GpuTensor out = new GpuTensor(device, targetShape);
        int totalElements = out.elements();

        TensorKey targetShapeKey = new TensorKey(Usage.OTHER, targetShape);
        TensorKey targetFlatStridesKey = new TensorKey(Usage.OTHER, targetFlatStrides);
        TensorKey alignedSrcShapeKey = new TensorKey(Usage.OTHER, alignedSrcShape);
        TensorKey alignedSrcStridesKey = new TensorKey(Usage.OTHER, alignedSrcStrides);

        ComputeBuffer targetShapeBuffer = device.acquire(targetShapeKey, () -> device.createBuffer(targetShape));
        ComputeBuffer targetFlatStridesBuffer = device.acquire(targetFlatStridesKey, () -> device.createBuffer(targetFlatStrides));
        ComputeBuffer alignedSrcShapeBuffer = device.acquire(alignedSrcShapeKey, () -> device.createBuffer(alignedSrcShape));
        ComputeBuffer alignedSrcStridesBuffer = device.acquire(alignedSrcStridesKey, () -> device.createBuffer(alignedSrcStrides));

        try (GpuContext.QueueHandle qh = GpuContext.getOrCreateQueue(device)) {
            KernelFactory.create(device, "broadcast_to")
                .buffer(dataBuffer)
                .buffer(out.dataBuffer)
                .buffer(targetShapeBuffer)
                .buffer(targetFlatStridesBuffer)
                .buffer(alignedSrcShapeBuffer)
                .buffer(alignedSrcStridesBuffer)
                .intVal(targetRank)
                .intVal(totalElements)
                .launch(qh.queue(), Math.max(1, totalElements));
        } catch (Throwable e) {
            throw new RuntimeException("Failed to launch broadcast kernel", e);
        }

        return out;
    }

    @Override
    public Tensor copy() {
        boolean contiguous = Arrays.equals(strides, Tensors.computeStrides(shape));

        GpuTensor result = new GpuTensor(device, shape);

        if (contiguous) {
            dataBuffer.copyInto(result.dataBuffer, device.queue());
            return result;
        }

        TensorKey stridesKey = new TensorKey(Usage.OTHER, strides);
        TensorKey shapeKey = new TensorKey(Usage.OTHER, shape);

        ComputeBuffer stridesBuffer = device.acquire(stridesKey, () -> device.createBuffer(strides));
        ComputeBuffer shapeBuffer = device.acquire(shapeKey, () -> device.createBuffer(shape));

        try (GpuContext.QueueHandle qh = GpuContext.getOrCreateQueue(device)) {
            KernelFactory.create(device, "copy_strided")
                .buffer(dataBuffer)
                .buffer(result.dataBuffer)
                .buffer(stridesBuffer)
                .buffer(shapeBuffer)
                .intVal(shape.length)
                .intVal(size)
                .launch(qh.queue(), Math.max(1, size));
        } catch (Throwable e) {
            throw new RuntimeException("Failed to copy tensor on the device", e);
        }

        return result;
    }

    private float[] contiguousData() {
        int size = elements();
        float[] raw = new float[size];
        dataBuffer.get(raw);

        float[] contiguous = new float[size];

        if (Arrays.equals(strides, Tensors.computeStrides(shape))) {
            System.arraycopy(raw, 0, contiguous, 0, size);
            return contiguous;
        }

        int rank = shape.length;
        int[] indices = new int[rank];

        for (int linear = 0; linear < size; linear++) {
            int offset = 0;
            for (int d = 0; d < rank; d++) {
                offset += indices[d] * strides[d];
            }

            contiguous[linear] = raw[offset];

            for (int d = rank - 1; d >= 0; d--) {
                if (++indices[d] < shape[d]) break;
                indices[d] = 0;
            }
        }

        return contiguous;
    }

    private static ComputeBuffer unsupportedRawBuffer() {
        throw new UnsupportedOperationException(
            "Raw native memory handles are not available in the Silicon GPU backend"
        );
    }
}
