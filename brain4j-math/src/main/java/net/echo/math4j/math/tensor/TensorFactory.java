package net.echo.math4j.math.tensor;

import net.echo.math4j.math.tensor.impl.TensorCPU;
import net.echo.math4j.math.tensor.impl.TensorGPU;
import net.echo.math4j.math.vector.Vector;

public class TensorFactory {
    
    private static boolean useGPU;

    public static void forceCPU() {
        useGPU = false;
    }
    
    public static void useGPUIfAvailable() {
        try {
            useGPU = TensorGPU.isGpuAvailable();
        } catch (Throwable t) {
            System.err.println("GPU acceleration not available: " + t.getMessage());
        }
    }
    
    public static boolean isUsingGPU() {
        return useGPU;
    }
    
    public static Tensor create(int... shape) {
        return useGPU ? new TensorGPU(shape) : new TensorCPU(shape);
    }
    
    public static Tensor of(int[] shape, float... data) {
        return useGPU ? TensorGPU.of(shape, data) : TensorCPU.of(shape, data);
    }
    
    public static Tensor of(int[] shape, double... data) {
        if (useGPU) {
            float[] floatData = new float[data.length];

            for (int i = 0; i < data.length; i++) {
                floatData[i] = (float) data[i];
            }

            return TensorGPU.of(shape, floatData);
        } else {
            return TensorCPU.of(shape, data);
        }
    }
    
    public static Tensor vector(float... data) {
        return of(new int[]{data.length}, data);
    }
    
    public static Tensor vector(Vector data) {
        if (useGPU) {
            int size = data.size();
            float[] floatData = new float[size];
            
            for (int i = 0; i < size; i++) {
                floatData[i] = data.get(i);
            }
            
            return TensorGPU.of(new int[]{size}, floatData);
        } else {
            return TensorCPU.vector(data);
        }
    }
    
    public static Tensor matrix(int rows, int cols, float... data) {
        return of(new int[]{rows, cols}, data);
    }
    
    public static Tensor zeros(int... shape) {
        return useGPU ? new TensorGPU(shape).fill(0.0) : TensorCPU.zeros(shape);
    }
    
    public static Tensor ones(int... shape) {
        return useGPU ? new TensorGPU(shape).fill(1.0) : TensorCPU.ones(shape);
    }
    
    public static Tensor random(int... shape) {
        return useGPU ?
            new TensorGPU(shape).fill(Math::random) :
            TensorCPU.random(shape);
    }

    public static Tensor random(long seed, int... shape) {
        Tensor cpuTensor = TensorCPU.random(seed, shape);
        return useGPU ? TensorGPU.fromTensor(cpuTensor) : cpuTensor;
    }
    
    public static Tensor uniform(double lowerBound, double upperBound, int... shape) {
        Tensor cpuTensor = TensorCPU.uniform(lowerBound, upperBound, shape);
        return useGPU ? TensorGPU.fromTensor(cpuTensor) : cpuTensor;
    }

    public static Tensor random(double mean, double standardDeviation, int... shape) {
        Tensor cpuTensor = TensorCPU.randn(mean, standardDeviation, shape);
        return useGPU ? TensorGPU.fromTensor(cpuTensor) : cpuTensor;
    }
    
    public static Tensor toGPU(TensorCPU tensor) {
        if (useGPU && !(tensor instanceof TensorGPU)) {
            return TensorGPU.fromTensor(tensor);
        }
        return tensor;
    }
} 