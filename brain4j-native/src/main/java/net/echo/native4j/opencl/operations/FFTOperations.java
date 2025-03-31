package net.echo.native4j.opencl.operations;

import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.impl.TensorGPU;
import net.echo.native4j.opencl.OpenCLContext;
import org.jocl.*;

import static org.jocl.CL.*;

public class FFTOperations {

    public static Tensor fft1D(Tensor input, boolean inverse) {
        OpenCLContext context = OpenCLContext.getInstance();
        
        if (!context.isInitialized()) {
            throw new RuntimeException("OpenCL not initialized");
        }
        
        if (input.dimension() != 1) {
            throw new IllegalArgumentException("Input tensor must be 1D for 1D FFT");
        }
        
        int inputSize = input.shape()[0];
        int fftSize = nextPowerOf2(inputSize);
        
        float[] inputData = input.toArray();
        float[] resultData = new float[2 * fftSize];
        
        cl_mem memInput = null;
        cl_mem memOutput = null;
        
        try {
            memInput = clCreateBuffer(context.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * inputData.length, Pointer.to(inputData), null);
            
            memOutput = clCreateBuffer(context.getContext(), CL_MEM_WRITE_ONLY,
                    (long) Sizeof.cl_float * resultData.length, null, null);
            
            cl_kernel kernel = context.getKernel("fft1d");
            
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(memInput));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(memOutput));
            clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[] { inputSize }));
            clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] { fftSize }));
            clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[] { inverse ? -1 : 1 }));
            
            long[] globalWorkSize = new long[] { fftSize };
            
            try {
                long[] localWorkSize = getOptimalWorkgroupSize(fftSize);
                clEnqueueNDRangeKernel(context.getCommandQueue(), kernel, 1, null,
                        globalWorkSize, localWorkSize, 0, null, null);
            } catch (Exception e) {
                clEnqueueNDRangeKernel(context.getCommandQueue(), kernel, 1, null,
                        globalWorkSize, null, 0, null, null);
            }
            
            clEnqueueReadBuffer(context.getCommandQueue(), memOutput, CL_TRUE, 0,
                    (long) Sizeof.cl_float * resultData.length, Pointer.to(resultData),
                    0, null, null);
            
            return TensorGPU.of(new int[] { 2, fftSize }, resultData);
        } finally {
            if (memInput != null) clReleaseMemObject(memInput);
            if (memOutput != null) clReleaseMemObject(memOutput);
        }
    }
    
    public static Tensor fft2D(Tensor input, boolean inverse) {
        OpenCLContext context = OpenCLContext.getInstance();
        
        if (!context.isInitialized()) {
            throw new RuntimeException("OpenCL not initialized");
        }
        
        if (input.dimension() != 2) {
            throw new IllegalArgumentException("Input tensor must be 2D for 2D FFT");
        }
        
        int[] shape = input.shape();
        int rows = shape[0];
        int cols = shape[1];
        
        int fftRows = nextPowerOf2(rows);
        int fftCols = nextPowerOf2(cols);
        
        float[] inputData = input.toArray();
        float[] resultData = new float[2 * fftRows * fftCols];
        
        cl_mem memInput = null;
        cl_mem memOutput = null;
        
        try {
            memInput = clCreateBuffer(context.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * inputData.length, Pointer.to(inputData), null);
            
            memOutput = clCreateBuffer(context.getContext(), CL_MEM_WRITE_ONLY,
                    (long) Sizeof.cl_float * resultData.length, null, null);
            
            cl_kernel kernel = context.getKernel("fft2d");
            
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(memInput));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(memOutput));
            clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[] { rows }));
            clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] { cols }));
            clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[] { inverse ? -1 : 1 }));
            
            long[] globalWorkSize = new long[] { fftRows, fftCols };
            
            try {
                long[] localWorkSize = getOptimalWorkgroupSize2D(fftRows, fftCols);
                clEnqueueNDRangeKernel(context.getCommandQueue(), kernel, 2, null,
                        globalWorkSize, localWorkSize, 0, null, null);
            } catch (Exception e) {
                clEnqueueNDRangeKernel(context.getCommandQueue(), kernel, 2, null,
                        globalWorkSize, null, 0, null, null);
            }
            
            clEnqueueReadBuffer(context.getCommandQueue(), memOutput, CL_TRUE, 0,
                    (long) Sizeof.cl_float * resultData.length, Pointer.to(resultData),
                    0, null, null);
            
            return TensorGPU.of(new int[] { 2, fftRows, fftCols }, resultData);
        } finally {
            if (memInput != null) clReleaseMemObject(memInput);
            if (memOutput != null) clReleaseMemObject(memOutput);
        }
    }
    
    public static Tensor complexPointwiseMul(Tensor a, Tensor b) {
        OpenCLContext context = OpenCLContext.getInstance();
        
        if (!context.isInitialized()) {
            throw new RuntimeException("OpenCL not initialized");
        }
        
        if (a.dimension() != b.dimension() || 
            (a.dimension() != 2 && a.dimension() != 3)) {
            throw new IllegalArgumentException("Tensors must have the same dimension (2 or 3) for complex multiplication");
        }
        
        int[] shapeA = a.shape();
        int[] shapeB = b.shape();
        
        if (a.dimension() == 2) {
            if (shapeA[0] != 2 || shapeB[0] != 2 || shapeA[1] != shapeB[1]) {
                throw new IllegalArgumentException("Invalid shapes for 1D complex multiplication");
            }
        } else {
            if (shapeA[0] != 2 || shapeB[0] != 2 || 
                shapeA[1] != shapeB[1] || shapeA[2] != shapeB[2]) {
                throw new IllegalArgumentException("Invalid shapes for 2D complex multiplication");
            }
        }
        
        int elements;
        if (a.dimension() == 2) {
            elements = shapeA[1];
        } else {
            elements = shapeA[1] * shapeA[2];
        }
        
        float[] dataA = a.toArray();
        float[] dataB = b.toArray();
        float[] resultData = new float[2 * elements];
        
        cl_mem memA = null;
        cl_mem memB = null;
        cl_mem memC = null;
        
        try {
            memA = clCreateBuffer(context.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * dataA.length, Pointer.to(dataA), null);
            memB = clCreateBuffer(context.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * dataB.length, Pointer.to(dataB), null);
            
            memC = clCreateBuffer(context.getContext(), CL_MEM_WRITE_ONLY,
                    (long) Sizeof.cl_float * resultData.length, null, null);
            
            cl_kernel kernel = context.getKernel("complex_pointwise_mul");
            
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(memA));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(memB));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(memC));
            clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] { elements }));
            
            long[] globalWorkSize = new long[] { elements };
            
            try {
                long[] localWorkSize = getOptimalWorkgroupSize(elements);
                clEnqueueNDRangeKernel(context.getCommandQueue(), kernel, 1, null,
                        globalWorkSize, localWorkSize, 0, null, null);
            } catch (Exception e) {
                clEnqueueNDRangeKernel(context.getCommandQueue(), kernel, 1, null,
                        globalWorkSize, null, 0, null, null);
            }
            
            clEnqueueReadBuffer(context.getCommandQueue(), memC, CL_TRUE, 0,
                    (long) Sizeof.cl_float * resultData.length, Pointer.to(resultData),
                    0, null, null);
            
            if (a.dimension() == 2) {
                return TensorGPU.of(new int[] { 2, elements }, resultData);
            } else {
                return TensorGPU.of(new int[] { 2, shapeA[1], shapeA[2] }, resultData);
            }
        } finally {
            if (memA != null) clReleaseMemObject(memA);
            if (memB != null) clReleaseMemObject(memB);
            if (memC != null) clReleaseMemObject(memC);
        }
    }

    private static int nextPowerOf2(int n) {
        if (n <= 0) return 1;
        n--;
        n |= n >>> 1;
        n |= n >>> 2;
        n |= n >>> 4;
        n |= n >>> 8;
        n |= n >>> 16;
        return n + 1;
    }

    private static long[] getOptimalWorkgroupSize(long globalSize) {
        int size = 1;
        while (size < 128 && size * 2 <= 1024) {
            size *= 2;
        }
        return new long[] { Math.min(size, globalSize) };
    }
    
    private static long[] getOptimalWorkgroupSize2D(long globalSizeX, long globalSizeY) {
        int sizeX = 16;
        int sizeY = 16;
        
        while (sizeX * sizeY > 256) {
            if (sizeX > sizeY) {
                sizeX /= 2;
            } else {
                sizeY /= 2;
            }
        }
        
        sizeX = nextPowerOf2(sizeX);
        sizeY = nextPowerOf2(sizeY);
        
        return new long[] { 
            Math.min(sizeX, globalSizeX),
            Math.min(sizeY, globalSizeY)
        };
    }
} 