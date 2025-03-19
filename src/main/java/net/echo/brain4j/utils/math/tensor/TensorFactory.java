package net.echo.brain4j.utils.math.tensor;

import net.echo.brain4j.utils.math.vector.Vector;

public class TensorFactory {
    
    private static boolean useGPU = false;
    
    static {
        try {
            useGPU = TensorGPU.isGpuAvailable();
        } catch (Throwable t) {
            System.err.println("GPU acceleration not available: " + t.getMessage());
            useGPU = false;
        }
    }

    public static void forceCPU() {
        useGPU = false;
    }
    
    public static void useGPUIfAvailable() {
        try {
            useGPU = TensorGPU.isGpuAvailable();
        } catch (Throwable t) {
            useGPU = false;
            System.err.println("GPU acceleration not available: " + t.getMessage());
        }
    }
    
    public static boolean isUsingGPU() {
        return useGPU;
    }
    
    public static Tensor create(int... shape) {
        return useGPU ? new TensorGPU(shape) : new Tensor(shape);
    }
    
    public static Tensor of(int[] shape, float... data) {
        return useGPU ? TensorGPU.of(shape, data) : Tensor.of(shape, data);
    }
    
    public static Tensor of(int[] shape, double... data) {
        if (useGPU) {
            float[] floatData = new float[data.length];
            for (int i = 0; i < data.length; i++) {
                floatData[i] = (float) data[i];
            }
            return TensorGPU.of(shape, floatData);
        } else {
            return Tensor.of(shape, data);
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
            return Tensor.vector(data);
        }
    }
    
    public static Tensor matrix(int rows, int cols, float... data) {
        return of(new int[]{rows, cols}, data);
    }
    
    public static Tensor zeros(int... shape) {
        return useGPU ? new TensorGPU(shape).fill(0.0) : Tensor.zeros(shape);
    }
    
    public static Tensor ones(int... shape) {
        return useGPU ? new TensorGPU(shape).fill(1.0) : Tensor.ones(shape);
    }
    
    public static Tensor random(int... shape) {
        return useGPU ? 
            new TensorGPU(shape).fill(Math::random) : 
            Tensor.random(shape);
    }

    public static Tensor random(long seed, int... shape) {
        Tensor cpuTensor = Tensor.random(seed, shape);
        return useGPU ? TensorGPU.fromTensor(cpuTensor) : cpuTensor;
    }
    
    public static Tensor uniform(double lowerBound, double upperBound, int... shape) {
        Tensor cpuTensor = Tensor.uniform(lowerBound, upperBound, shape);
        return useGPU ? TensorGPU.fromTensor(cpuTensor) : cpuTensor;
    }

    public static Tensor randn(double mean, double stddev, int... shape) {
        Tensor cpuTensor = Tensor.randn(mean, stddev, shape);
        return useGPU ? TensorGPU.fromTensor(cpuTensor) : cpuTensor;
    }
    
    public static Tensor toGPU(Tensor tensor) {
        if (useGPU && !(tensor instanceof TensorGPU)) {
            return TensorGPU.fromTensor(tensor);
        }
        return tensor;
    }
} 