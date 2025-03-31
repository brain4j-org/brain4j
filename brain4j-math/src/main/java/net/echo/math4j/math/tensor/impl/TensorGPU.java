package net.echo.math4j.math.tensor.impl;

import net.echo.math4j.math.tensor.Tensor;
import net.echo.native4j.opencl.OpenCLContext;
import net.echo.native4j.opencl.operations.MatrixOperations;
import net.echo.native4j.opencl.operations.TensorOperations;
import net.echo.native4j.opencl.operations.ConvolutionOperations;


public class TensorGPU extends TensorCPU {

    private static boolean INITIALIZED;
    
    static {
        try {
            OpenCLContext context = OpenCLContext.getInstance();
            INITIALIZED = context.isInitialized();
        } catch (Exception e) {
            System.err.println("GPU acceleration not available: " + e.getMessage());
            INITIALIZED = false;
        }
    }
    
    public TensorGPU(int... shape) {
        super(shape);
    }

    public static Tensor fromTensor(Tensor tensor) {
        Tensor gpuTensor = new TensorGPU(tensor.shape());
        int[] shape = tensor.shape();
        
        for (int i = 0; i < tensor.elements(); i++) {
            if (shape.length == 1) {
                gpuTensor.set(tensor.get(i), i);
            } else {
                int[] indices = linearToMultiDimIndices(i, shape);
                float value = tensor.get(indices);

                gpuTensor.set(value, indices);
            }
        }

        return gpuTensor;
    }

    public static Tensor of(int[] shape, float... data) {
        TensorGPU tensor = new TensorGPU(shape);

        for (int i = 0; i < data.length; i++) {
            tensor.getData().set(i, data[i]);
        }

        return tensor;
    }
    
    public static Tensor matrix(int rows, int cols, float... data) {
        return of(new int[]{rows, cols}, data);
    }

    public static void releaseGPUResources() {
        if (INITIALIZED) {
            try {
                OpenCLContext.getInstance().releaseResources();
                INITIALIZED = false;
            } catch (Exception e) {
                System.err.println("Error releasing GPU resources: " + e.getMessage());
            }
        }
    }

    public static void reinitializeGPU() {        
        if (!INITIALIZED) {
            try {
                INITIALIZED = OpenCLContext.getInstance().reinitialize();
            } catch (Exception e) {
                System.err.println("Failed to reinitialize GPU: " + e.getMessage());
            }
        }
    }

    private static int[] linearToMultiDimIndices(int linearIndex, int[] shape) {
        int[] indices = new int[shape.length];
        int remainingIndex = linearIndex;

        for (int i = shape.length - 1; i >= 0; i--) {
            indices[i] = remainingIndex % shape[i];
            remainingIndex /= shape[i];
        }

        return indices;
    }

    public static boolean isGpuAvailable() {
        return INITIALIZED;
    }
    
    @Override
    public Tensor matmul(Tensor other) {
        if (!INITIALIZED) {
            return super.matmul(other); 
        }
        
        try {
            return MatrixOperations.matmul(this, other);
        } catch (Exception e) {
            System.err.println("GPU matmul failed, falling back to CPU: " + e.getMessage());
            return super.matmul(other);
        }
    }
    
    @Override
    public Tensor add(Tensor other) {
        if (!INITIALIZED) {
            return super.add(other); 
        }
        
        try {
            return TensorOperations.elementWiseAdd(this, other);
        } catch (Exception e) {
            System.err.println("GPU element-wise addition failed: " + e.getMessage());
            return super.add(other);
        }
    }
    
    @Override
    public Tensor mul(Tensor other) {
        if (!INITIALIZED) {
            return super.mul(other); 
        }
        
        try {
            return TensorOperations.elementWiseMul(this, other);
        } catch (Exception e) {
            System.err.println("GPU mul failed, falling back to CPU: " + e.getMessage());
            return super.mul(other);
        }
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor convolve(Tensor kernel) {
        if (!INITIALIZED) {
            return super.convolve(kernel);
        }
        
        try {
            return ConvolutionOperations.convolve(this, kernel);
        } catch (Exception e) {
            System.err.println("Error in GPU convolution: " + e.getMessage());
            return super.convolve(kernel);
        }
    }

    @Override
    public Tensor softmax() {
        if (!INITIALIZED) {
            return super.softmax();
        }
        
        Tensor cpuTensor = cpu();
        Tensor result = cpuTensor.softmax();
        return result.gpu();
    }

    @Override
    public Tensor gpu() {
        return this;
    }

    @Override
    public Tensor cpu() {
        return TensorCPU.of(shape(), getData());
    }
}