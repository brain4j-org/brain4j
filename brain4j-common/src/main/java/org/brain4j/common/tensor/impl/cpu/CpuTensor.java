package org.brain4j.common.tensor.impl.cpu;

import org.brain4j.common.device.Device;
import org.brain4j.common.tensor.Tensor;
import org.brain4j.common.tensor.TensorImplBase;
import org.brain4j.common.tensor.impl.cpu.matmul.MatmulProvider;
import org.brain4j.common.tensor.impl.cpu.matmul.NormalMatmulProvider;
import org.brain4j.common.tensor.impl.cpu.matmul.SimdMatmulProvider;
import org.brain4j.common.tensor.impl.gpu.GpuTensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.concurrent.ForkJoinPool;

public class CpuTensor extends TensorImplBase {

    private static final Logger logger = LoggerFactory.getLogger(CpuTensor.class);

    private static ForkJoinPool pool;
    private static MatmulProvider matmulProvider;

    public static void initialize() {
        Optional<Module> module = ModuleLayer.boot().findModule("jdk.incubator.vector");

        pool = ForkJoinPool.commonPool();

        if (module.isPresent()) {
            matmulProvider = new SimdMatmulProvider();
        } else {
            logger.warn("The Vector incubator API is not available. It's recommended to use for better performance.");
            logger.warn("For more information consult this guide: https://github.com/brain4j-org/brain4j/wiki/Using-SIMD");

            matmulProvider = new NormalMatmulProvider();
        }
    }

    public CpuTensor(int[] shape, float... data) {
        if (data.length == 0) {
            data = new float[computeSize(shape)];
        }

        this.data = data;
        this.shape = shape;
        this.strides = computeStrides(shape);
    }

    @Override
    public Tensor to(Device device) {
        if (device == null) {
            return this;
        }

        return new GpuTensor(device, shape, data);
    }

    @Override
    public Tensor add(Tensor other) {
        if (!(other instanceof CpuTensor)) {
            return add(other.cpu());
        }
        return super.add(other);
    }

    @Override
    public Tensor sub(Tensor other) {
        if (!(other instanceof CpuTensor)) {
            return sub(other.cpu());
        }
        return super.sub(other);
    }

    @Override
    public Tensor mul(Tensor other) {
        if (!(other instanceof CpuTensor)) {
            return mul(other.cpu());
        }
        return super.mul(other);
    }

    @Override
    public Tensor div(Tensor other) {
        if (!(other instanceof CpuTensor)) {
            return div(other.cpu());
        }
        return super.div(other);
    }

    @Override
    public Tensor pow(Tensor other) {
        if (!(other instanceof CpuTensor)) {
            return pow(other.cpu());
        }
        return super.pow(other);
    }

    @Override
    public Tensor matmul(Tensor other) {
        if (matmulProvider == null) {
            initialize();
        }

        int[] shapeA = this.shape;
        int[] shapeB = other.shape();

        if (shapeA.length < 2 || shapeB.length < 2) {
            throw new IllegalArgumentException("Matrix multiplication requires at least 2D tensors!");
        }

        int rankA = shapeA.length;
        int rankB = shapeB.length;

        int m = shapeA[rankA - 2];
        int n = shapeA[rankA - 1];

        int k = shapeB[rankB - 2];
        int p = shapeB[rankB - 1];

        if (n != k) {
            throw new IllegalArgumentException("Inner dimensions must match: " + n + " != " + k);
        }

        int maxBatchDims = Math.max(rankA, rankB) - 2;
        int[] batchShape = new int[maxBatchDims];

        for (int i = 0; i < maxBatchDims; i++) {
            int dimA = (i < rankA - 2) ? shapeA[i + rankA - 2 - maxBatchDims] : 1;
            int dimB = (i < rankB - 2) ? shapeB[i + rankB - 2 - maxBatchDims] : 1;

            if (dimA != dimB && dimA != 1 && dimB != 1) {
                throw new IllegalArgumentException(
                    "Cannot broadcast batch dimensions: " + dimA + " vs " + dimB + " at batch dim index " + i
                );
            }

            batchShape[i] = Math.max(dimA, dimB);
        }

        int[] resultShape = new int[batchShape.length + 2];

        System.arraycopy(batchShape, 0, resultShape, 0, batchShape.length);
        resultShape[resultShape.length - 2] = m;
        resultShape[resultShape.length - 1] = p;

        Tensor result = new CpuTensor(resultShape);

        float[] A = this.data();
        float[] B = other.data();
        float[] C = result.data();

        int batchA = 1;

        for (int i = 0; i < rankA - 2; i++) {
            batchA *= shapeA[i];
        }

        int batchB = 1;

        for (int i = 0; i < rankB - 2; i++) {
            batchB *= shapeB[i];
        }

        int batchCount = 1;

        for (int d : batchShape) {
            batchCount *= d;
        }

        matmulProvider.multiply(pool, batchCount, m, n, p, A, B, C, batchA, batchB);

        return result;
    }
}
