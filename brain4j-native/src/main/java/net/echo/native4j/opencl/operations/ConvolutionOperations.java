package net.echo.native4j.opencl.operations;

import net.echo.math4j.math.constants.Constants;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.impl.TensorGPU;
import net.echo.native4j.opencl.OpenCLContext;
import org.jocl.*;

import static org.jocl.CL.*;

public class ConvolutionOperations {

    public static Tensor convolve(Tensor input, Tensor kernel) {
        if (input.dimension() == 1 && kernel.dimension() == 1) {
            return convolve1D(input, kernel);
        } else if (input.dimension() == 2 && kernel.dimension() == 2) {
            return convolve2D(input, kernel);
        } else {
            throw new IllegalArgumentException(
                "Convolution supported only for 1D or 2D tensors, dimensions: " +
                input.dimension() + " and " + kernel.dimension());
        }
    }

    public static Tensor convolve1D(Tensor input, Tensor kernel) {
        OpenCLContext context = OpenCLContext.getInstance();
        
        if (!context.isInitialized()) {
            throw new RuntimeException("OpenCL not initialized");
        }
        
        int inputSize = input.shape()[0];
        int kernelSize = kernel.shape()[0];
        
        int outputSize = inputSize;  
        int totalPadding = kernelSize - 1;
        int paddingLeft = totalPadding / 2;
        
        boolean useFFT = kernelSize > Constants.FFT_THRESHOLD;
        
        float[] inputData = input.toArray();
        float[] kernelData = kernel.toArray();
        float[] resultData = new float[outputSize];
        
        cl_mem memInput = null;
        cl_mem memKernel = null;
        cl_mem memOutput = null;
        cl_mem memBuffer1 = null;
        cl_mem memBuffer2 = null;
        cl_mem memBuffer3 = null;
        
        try {
            memInput = clCreateBuffer(context.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * inputData.length, Pointer.to(inputData), null);
            memKernel = clCreateBuffer(context.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * kernelData.length, Pointer.to(kernelData), null);
            memOutput = clCreateBuffer(context.getContext(), CL_MEM_WRITE_ONLY,
                    (long) Sizeof.cl_float * resultData.length, null, null);
            
            if (!useFFT) {
                cl_kernel directKernel = context.getKernel("convolve1d_direct");
                
                clSetKernelArg(directKernel, 0, Sizeof.cl_mem, Pointer.to(memInput));
                clSetKernelArg(directKernel, 1, Sizeof.cl_mem, Pointer.to(memKernel));
                clSetKernelArg(directKernel, 2, Sizeof.cl_mem, Pointer.to(memOutput));
                clSetKernelArg(directKernel, 3, Sizeof.cl_int, Pointer.to(new int[] { inputSize }));
                clSetKernelArg(directKernel, 4, Sizeof.cl_int, Pointer.to(new int[] { kernelSize }));
                clSetKernelArg(directKernel, 5, Sizeof.cl_int, Pointer.to(new int[] { outputSize }));
                clSetKernelArg(directKernel, 6, Sizeof.cl_int, Pointer.to(new int[] { paddingLeft }));
                
                long[] globalWorkSize = new long[] { outputSize };
                
                try {
                    long[] localWorkSize = getOptimalWorkgroupSize(outputSize);
                    clEnqueueNDRangeKernel(context.getCommandQueue(), directKernel, 1, null,
                            globalWorkSize, localWorkSize, 0, null, null);
                } catch (Exception e) {
                    clEnqueueNDRangeKernel(context.getCommandQueue(), directKernel, 1, null,
                            globalWorkSize, null, 0, null, null);
                }
            } else {
                int fftSize = nextPowerOf2(inputSize + kernelSize - 1);
                
                memBuffer1 = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                        (long) Sizeof.cl_float * 2 * fftSize, null, null);
                memBuffer2 = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                        (long) Sizeof.cl_float * 2 * fftSize, null, null);
                memBuffer3 = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                        (long) Sizeof.cl_float * 2 * fftSize, null, null);
                
                cl_kernel fftKernel = context.getKernel("convolve1d_fft");
                
                clSetKernelArg(fftKernel, 0, Sizeof.cl_mem, Pointer.to(memInput));
                clSetKernelArg(fftKernel, 1, Sizeof.cl_mem, Pointer.to(memKernel));
                clSetKernelArg(fftKernel, 2, Sizeof.cl_mem, Pointer.to(memOutput));
                clSetKernelArg(fftKernel, 3, Sizeof.cl_mem, Pointer.to(memBuffer1));
                clSetKernelArg(fftKernel, 4, Sizeof.cl_mem, Pointer.to(memBuffer2));
                clSetKernelArg(fftKernel, 5, Sizeof.cl_mem, Pointer.to(memBuffer3));
                clSetKernelArg(fftKernel, 6, Sizeof.cl_int, Pointer.to(new int[] { inputSize }));
                clSetKernelArg(fftKernel, 7, Sizeof.cl_int, Pointer.to(new int[] { kernelSize }));
                clSetKernelArg(fftKernel, 8, Sizeof.cl_int, Pointer.to(new int[] { fftSize }));
                clSetKernelArg(fftKernel, 9, Sizeof.cl_int, Pointer.to(new int[] { outputSize }));
                clSetKernelArg(fftKernel, 10, Sizeof.cl_int, Pointer.to(new int[] { paddingLeft }));
                
                long[] globalWorkSize = new long[] { fftSize };
                
                try {
                    long[] localWorkSize = getOptimalWorkgroupSize(fftSize);
                    clEnqueueNDRangeKernel(context.getCommandQueue(), fftKernel, 1, null,
                            globalWorkSize, localWorkSize, 0, null, null);
                } catch (Exception e) {
                    clEnqueueNDRangeKernel(context.getCommandQueue(), fftKernel, 1, null,
                            globalWorkSize, null, 0, null, null);
                }
            }
            
            clEnqueueReadBuffer(context.getCommandQueue(), memOutput, CL_TRUE, 0,
                    (long) Sizeof.cl_float * resultData.length, Pointer.to(resultData),
                    0, null, null);
            
            return TensorGPU.of(new int[] { outputSize }, resultData);
        } finally {
            if (memInput != null) clReleaseMemObject(memInput);
            if (memKernel != null) clReleaseMemObject(memKernel);
            if (memOutput != null) clReleaseMemObject(memOutput);
            if (memBuffer1 != null) clReleaseMemObject(memBuffer1);
            if (memBuffer2 != null) clReleaseMemObject(memBuffer2);
            if (memBuffer3 != null) clReleaseMemObject(memBuffer3);
        }
    }
    
    public static Tensor convolve2D(Tensor input, Tensor kernel) {
        OpenCLContext context = OpenCLContext.getInstance();
        
        if (!context.isInitialized()) {
            throw new RuntimeException("OpenCL not initialized");
        }
        
        int[] inputShape = input.shape();
        int[] kernelShape = kernel.shape();
        
        int inputRows = inputShape[0];
        int inputCols = inputShape[1];
        int kernelRows = kernelShape[0];
        int kernelCols = kernelShape[1];
        
        int outputRows = inputRows;
        int outputCols = inputCols;
        
        int paddingTop = (kernelRows - 1) / 2;
        int paddingLeft = (kernelCols - 1) / 2;
        
        boolean useFFT = (kernelRows * kernelCols) > Constants.FFT_THRESHOLD;
        
        float[] inputData = input.toArray();
        float[] kernelData = kernel.toArray();
        float[] resultData = new float[outputRows * outputCols];
        
        cl_mem memInput = null;
        cl_mem memKernel = null;
        cl_mem memOutput = null;
        cl_mem memBuffer1 = null;
        cl_mem memBuffer2 = null;
        cl_mem memBuffer3 = null;
        
        try {
            memInput = clCreateBuffer(context.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * inputData.length, Pointer.to(inputData), null);
            memKernel = clCreateBuffer(context.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * kernelData.length, Pointer.to(kernelData), null);
            memOutput = clCreateBuffer(context.getContext(), CL_MEM_WRITE_ONLY,
                    (long) Sizeof.cl_float * resultData.length, null, null);
            
            if (!useFFT) {
                cl_kernel directKernel = context.getKernel("convolve2d_direct");
                
                clSetKernelArg(directKernel, 0, Sizeof.cl_mem, Pointer.to(memInput));
                clSetKernelArg(directKernel, 1, Sizeof.cl_mem, Pointer.to(memKernel));
                clSetKernelArg(directKernel, 2, Sizeof.cl_mem, Pointer.to(memOutput));
                clSetKernelArg(directKernel, 3, Sizeof.cl_int, Pointer.to(new int[] { inputRows }));
                clSetKernelArg(directKernel, 4, Sizeof.cl_int, Pointer.to(new int[] { inputCols }));
                clSetKernelArg(directKernel, 5, Sizeof.cl_int, Pointer.to(new int[] { kernelRows }));
                clSetKernelArg(directKernel, 6, Sizeof.cl_int, Pointer.to(new int[] { kernelCols }));
                clSetKernelArg(directKernel, 7, Sizeof.cl_int, Pointer.to(new int[] { outputRows }));
                clSetKernelArg(directKernel, 8, Sizeof.cl_int, Pointer.to(new int[] { outputCols }));
                clSetKernelArg(directKernel, 9, Sizeof.cl_int, Pointer.to(new int[] { paddingTop }));
                clSetKernelArg(directKernel, 10, Sizeof.cl_int, Pointer.to(new int[] { paddingLeft }));
                
                long[] globalWorkSize = new long[] { outputRows, outputCols };
                
                try {
                    long[] localWorkSize = getOptimalWorkgroupSize2D(outputRows, outputCols);
                    clEnqueueNDRangeKernel(context.getCommandQueue(), directKernel, 2, null,
                            globalWorkSize, localWorkSize, 0, null, null);
                } catch (Exception e) {
                    clEnqueueNDRangeKernel(context.getCommandQueue(), directKernel, 2, null,
                            globalWorkSize, null, 0, null, null);
                }
            } else {
                int fftRows = nextPowerOf2(inputRows + kernelRows - 1);
                int fftCols = nextPowerOf2(inputCols + kernelCols - 1);
                
                memBuffer1 = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                        (long) Sizeof.cl_float * 2 * fftRows * fftCols, null, null);
                memBuffer2 = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                        (long) Sizeof.cl_float * 2 * fftRows * fftCols, null, null);
                memBuffer3 = clCreateBuffer(context.getContext(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                        (long) Sizeof.cl_float * 2 * fftRows * fftCols, null, null);
                
                cl_kernel fft2dKernel = context.getKernel("fft2d");
                
                clSetKernelArg(fft2dKernel, 0, Sizeof.cl_mem, Pointer.to(memInput));
                clSetKernelArg(fft2dKernel, 1, Sizeof.cl_mem, Pointer.to(memBuffer1));
                clSetKernelArg(fft2dKernel, 2, Sizeof.cl_int, Pointer.to(new int[] { inputRows }));
                clSetKernelArg(fft2dKernel, 3, Sizeof.cl_int, Pointer.to(new int[] { inputCols }));
                clSetKernelArg(fft2dKernel, 4, Sizeof.cl_int, Pointer.to(new int[] { 1 })); // direction = 1 per FFT diretta
                
                long[] globalWorkSize = new long[] { fftRows, fftCols };
                
                try {
                    long[] localWorkSize = getOptimalWorkgroupSize2D(fftRows, fftCols);
                    clEnqueueNDRangeKernel(context.getCommandQueue(), fft2dKernel, 2, null,
                            globalWorkSize, localWorkSize, 0, null, null);
                } catch (Exception e) {
                    clEnqueueNDRangeKernel(context.getCommandQueue(), fft2dKernel, 2, null,
                            globalWorkSize, null, 0, null, null);
                }
                
                clSetKernelArg(fft2dKernel, 0, Sizeof.cl_mem, Pointer.to(memKernel));
                clSetKernelArg(fft2dKernel, 1, Sizeof.cl_mem, Pointer.to(memBuffer2));
                clSetKernelArg(fft2dKernel, 2, Sizeof.cl_int, Pointer.to(new int[] { kernelRows }));
                clSetKernelArg(fft2dKernel, 3, Sizeof.cl_int, Pointer.to(new int[] { kernelCols }));
                
                try {
                    long[] localWorkSize = getOptimalWorkgroupSize2D(fftRows, fftCols);
                    clEnqueueNDRangeKernel(context.getCommandQueue(), fft2dKernel, 2, null,
                            globalWorkSize, localWorkSize, 0, null, null);
                } catch (Exception e) {
                    clEnqueueNDRangeKernel(context.getCommandQueue(), fft2dKernel, 2, null,
                            globalWorkSize, null, 0, null, null);
                }
                
                cl_kernel pointwiseMulKernel = context.getKernel("complex_pointwise_mul");
                
                clSetKernelArg(pointwiseMulKernel, 0, Sizeof.cl_mem, Pointer.to(memBuffer1));
                clSetKernelArg(pointwiseMulKernel, 1, Sizeof.cl_mem, Pointer.to(memBuffer2));
                clSetKernelArg(pointwiseMulKernel, 2, Sizeof.cl_mem, Pointer.to(memBuffer3));
                clSetKernelArg(pointwiseMulKernel, 3, Sizeof.cl_int, Pointer.to(new int[] { fftRows * fftCols }));
                
                try {
                    long[] localWorkSize = getOptimalWorkgroupSize((long) fftRows * fftCols);
                    clEnqueueNDRangeKernel(context.getCommandQueue(), pointwiseMulKernel, 1, null,
                            new long[] {(long) fftRows * fftCols }, localWorkSize, 0, null, null);
                } catch (Exception e) {
                    clEnqueueNDRangeKernel(context.getCommandQueue(), pointwiseMulKernel, 1, null,
                            new long[] {(long) fftRows * fftCols }, null, 0, null, null);
                }
                
                clSetKernelArg(fft2dKernel, 0, Sizeof.cl_mem, Pointer.to(memBuffer3));
                clSetKernelArg(fft2dKernel, 1, Sizeof.cl_mem, Pointer.to(memOutput));
                clSetKernelArg(fft2dKernel, 2, Sizeof.cl_int, Pointer.to(new int[] { fftRows }));
                clSetKernelArg(fft2dKernel, 3, Sizeof.cl_int, Pointer.to(new int[] { fftCols }));
                clSetKernelArg(fft2dKernel, 4, Sizeof.cl_int, Pointer.to(new int[] { -1 })); // direction = -1 per IFFT
                
                try {
                    long[] localWorkSize = getOptimalWorkgroupSize2D(fftRows, fftCols);
                    clEnqueueNDRangeKernel(context.getCommandQueue(), fft2dKernel, 2, null,
                            globalWorkSize, localWorkSize, 0, null, null);
                } catch (Exception e) {
                    clEnqueueNDRangeKernel(context.getCommandQueue(), fft2dKernel, 2, null,
                            globalWorkSize, null, 0, null, null);
                }
                
                cl_kernel extractKernel = context.getKernel("convolve2d_fft_extract");
                
                clSetKernelArg(extractKernel, 0, Sizeof.cl_mem, Pointer.to(memOutput));
                clSetKernelArg(extractKernel, 1, Sizeof.cl_mem, Pointer.to(memOutput));
                clSetKernelArg(extractKernel, 2, Sizeof.cl_int, Pointer.to(new int[] { outputRows }));
                clSetKernelArg(extractKernel, 3, Sizeof.cl_int, Pointer.to(new int[] { outputCols }));
                clSetKernelArg(extractKernel, 4, Sizeof.cl_int, Pointer.to(new int[] { paddingTop }));
                clSetKernelArg(extractKernel, 5, Sizeof.cl_int, Pointer.to(new int[] { paddingLeft }));
                clSetKernelArg(extractKernel, 6, Sizeof.cl_int, Pointer.to(new int[] { fftCols }));
                
                try {
                    long[] localWorkSize = getOptimalWorkgroupSize2D(outputRows, outputCols);
                    clEnqueueNDRangeKernel(context.getCommandQueue(), extractKernel, 2, null,
                            new long[] { outputRows, outputCols }, localWorkSize, 0, null, null);
                } catch (Exception e) {
                    clEnqueueNDRangeKernel(context.getCommandQueue(), extractKernel, 2, null,
                            new long[] { outputRows, outputCols }, null, 0, null, null);
                }
            }
            
            clEnqueueReadBuffer(context.getCommandQueue(), memOutput, CL_TRUE, 0,
                    (long) Sizeof.cl_float * resultData.length, Pointer.to(resultData),
                    0, null, null);
            
            return TensorGPU.of(new int[] { outputRows, outputCols }, resultData);
        } finally {
            if (memInput != null) clReleaseMemObject(memInput);
            if (memKernel != null) clReleaseMemObject(memKernel);
            if (memOutput != null) clReleaseMemObject(memOutput);
            if (memBuffer1 != null) clReleaseMemObject(memBuffer1);
            if (memBuffer2 != null) clReleaseMemObject(memBuffer2);
            if (memBuffer3 != null) clReleaseMemObject(memBuffer3);
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
        while (size < Constants.OPTIMAL_WORKGROUP_SIZE && size * 2 <= Constants.MAX_WORKGROUP_SIZE) {
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