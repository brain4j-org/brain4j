package org.brain4j.math.tensor.impl;

import org.brain4j.math.device.DeviceType;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.TensorImplBase;
import org.brain4j.math.tensor.cpu.matmul.MatmulProvider;
import org.brain4j.math.tensor.cpu.matmul.NormalMatmulProvider;
import org.brain4j.math.tensor.cpu.matmul.SimdMatmulProvider;
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
    public Tensor to(DeviceType deviceType) {
        return switch (deviceType) {
            case CPU -> this;
            case GPU -> new GpuTensor(shape, data);
            default -> throw new IllegalArgumentException("Unsupported device type: " + deviceType);
        };
    }

    @Override
    public Tensor matmul(Tensor other) {
        if (matmulProvider == null) {
            throw new IllegalStateException(
                    "Matmul provider is not initialized. Make sure to call Brain4J.initialize(DeviceType) first."
            );
        }

        if (shape.length < 2 || other.shape().length < 2) {
            throw new IllegalArgumentException("Matrix multiplication requires at least 2D tensors!");
        }

        if (shape.length != other.shape().length) {
            throw new IllegalArgumentException(
                    "Dimensions do not match: " + shape.length + " != " + other.shape().length
            );
        }

        for (int i = 0; i < shape.length - 2; i++) {
            if (shape[i] != other.shape()[i]) {
                throw new IllegalArgumentException(
                        "Batch dimensions do not match at index " + i + ": " + shape[i] + " != " + other.shape()[i]
                );
            }
        }

        int dims = shape.length;

        int m = shape[dims - 2];
        int n = shape[dims - 1];

        int k = other.shape()[dims - 2];
        int p = other.shape()[dims - 1];

        if (n != k) {
            throw new IllegalArgumentException("Inner dimensions must match: " + n + " != " + k);
        }

        int[] resultShape = new int[dims];

        int batch = 1;
        for (int i = 0; i < dims - 2; i++) {
            resultShape[i] = shape[i];
            batch *= shape[i];
        }

        resultShape[dims - 2] = m;
        resultShape[dims - 1] = p;

        Tensor result = new CpuTensor(resultShape);

        float[] A = this.data();
        float[] B = other.data();
        float[] C = result.data();

        matmulProvider.multiply(batch, m, n, p, A, B, C, pool);
        return result;
    }
}
