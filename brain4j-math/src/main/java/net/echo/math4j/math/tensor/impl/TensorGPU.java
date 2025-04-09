package net.echo.math4j.math.tensor.impl;

import static net.echo.math4j.math.constants.Constants.*;

import net.echo.math4j.exceptions.NativeException;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.opencl.DeviceUtils;
import org.jocl.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.jocl.CL.*;

public class TensorGPU extends TensorCPU {

    private static boolean INITIALIZED;
    private static cl_device_id DEVICE_ID;
    private static cl_context CONTEXT;
    private static cl_command_queue COMMAND_QUEUE;
    private static cl_program MAIN_PROGRAM;
    private static cl_kernel MAT_MULT_KERNEL;
    private static cl_kernel ELEMENT_WISE_ADD_KERNEL;
    private static cl_kernel ELEMENT_WISE_MULT_KERNEL;
    
    private static cl_kernel CONVOLVE_1D_DIRECT_KERNEL;
    private static cl_kernel CONVOLVE_2D_DIRECT_KERNEL;
    
    private static cl_kernel FFT_1D_KERNEL;
    private static cl_kernel FFT_2D_KERNEL;
    private static cl_kernel FFT_2D_TRANSPOSE_KERNEL;
    private static cl_kernel COMPLEX_POINTWISE_MUL_KERNEL;
    private static cl_kernel CONVOLVE_1D_FFT_KERNEL;
    private static cl_kernel CONVOLVE_2D_FFT_EXTRACT_KERNEL;
    
    private static final String[] KERNEL_FILES = {
        "/kernels/common_definitions.cl",
        "/kernels/complex/complex_ops.cl",
        "/kernels/basic/tensor_ops.cl",
        "/kernels/transforms/bit_operations.cl",
        "/kernels/transforms/fft1d.cl",
        "/kernels/transforms/fft2d.cl",
        "/kernels/convolution/conv1d.cl",
        "/kernels/convolution/conv2d.cl"
    };
    
    static {
        try {
            INITIALIZED = initializeOpenCL();
        } catch (NativeException e) {
            System.err.println("GPU acceleration not available: " + e.getMessage());
        }
    }
    
    private static boolean initializeOpenCL() throws NativeException {
        try {
            CL.setExceptionsEnabled(true);
            
            cl_device_id device = DeviceUtils.findDevice(DeviceUtils.DeviceType.GPU);

            if (device == null) {
                System.out.println("No GPU device found, falling back to CPU");
                device = DeviceUtils.findDevice(DeviceUtils.DeviceType.CPU);
            }
            
            cl_device_id[] devices = {device};
            DEVICE_ID = device;
            CONTEXT = clCreateContext(null, 1, devices, null, null, null);
            COMMAND_QUEUE = clCreateCommandQueue(CONTEXT, device, 0, null);
            
            try {
                String combinedKernelSource = loadAndCombineKernels();
                
                MAIN_PROGRAM = clCreateProgramWithSource(CONTEXT, 1, 
                        new String[] { combinedKernelSource }, null, null);
                int buildResult = clBuildProgram(MAIN_PROGRAM, 0, null, "-cl-mad-enable -cl-fast-relaxed-math", null, null);
                
                if (buildResult != CL_SUCCESS) {
                    checkKernelStatus(MAIN_PROGRAM, DEVICE_ID, "main_program");
                    return false;
                }
                
                MAT_MULT_KERNEL = clCreateKernel(MAIN_PROGRAM, "matmul", null);
                ELEMENT_WISE_ADD_KERNEL = clCreateKernel(MAIN_PROGRAM, "element_wise_add", null);
                ELEMENT_WISE_MULT_KERNEL = clCreateKernel(MAIN_PROGRAM, "element_wise_mul", null);
                
                CONVOLVE_1D_DIRECT_KERNEL = clCreateKernel(MAIN_PROGRAM, "convolve1d_direct", null);
                CONVOLVE_2D_DIRECT_KERNEL = clCreateKernel(MAIN_PROGRAM, "convolve2d_direct", null);
                
                FFT_1D_KERNEL = clCreateKernel(MAIN_PROGRAM, "fft1d", null);
                FFT_2D_KERNEL = clCreateKernel(MAIN_PROGRAM, "fft2d", null);
                FFT_2D_TRANSPOSE_KERNEL = clCreateKernel(MAIN_PROGRAM, "fft2d_transpose", null);
                COMPLEX_POINTWISE_MUL_KERNEL = clCreateKernel(MAIN_PROGRAM, "complex_pointwise_mul", null);
                CONVOLVE_1D_FFT_KERNEL = clCreateKernel(MAIN_PROGRAM, "convolve1d_fft", null);
                CONVOLVE_2D_FFT_EXTRACT_KERNEL = clCreateKernel(MAIN_PROGRAM, "convolve2d_fft_extract", null);

                System.out.println("GPU acceleration enabled using device: " + DeviceUtils.getDeviceName());
                return true;
            } catch (Exception e) {
                throw new NativeException("Exception caught loading or compiling kernels! " + e.getMessage());
            }
        } catch (Exception e) {
            throw new NativeException("Failed to initialize OpenCL! " + e.getMessage());
        }
    }
    
    private static String loadAndCombineKernels() throws NativeException {
        StringBuilder combined = new StringBuilder();
        combined.append("// Brain4J OpenCL Kernels - automatically generated\n\n");
        
        Map<String, Boolean> includedFiles = new HashMap<>();
        
        for (String filePath : KERNEL_FILES) {
            String content = loadKernelSource(filePath);
            
            content = removeIncludes(content);
            
            combined.append("\n// ===== BEGIN ").append(filePath).append(" =====\n\n");
            combined.append(content);
            combined.append("\n// ===== END ").append(filePath).append(" =====\n\n");
            
            includedFiles.put(filePath, true);
        }
        
        return combined.toString();
    }
    
    private static String removeIncludes(String source) {
        StringBuilder result = new StringBuilder();
        String[] lines = source.split("\n");
        
        for (String line : lines) {
            if (!line.trim().startsWith("#include")) {
                result.append(line).append("\n");
            }
        }
        
        return result.toString();
    }
    
    private static String loadKernelSource(String resourceName) throws NativeException {
        InputStream inputStream = TensorGPU.class.getResourceAsStream(resourceName);
        
        if (inputStream == null) {
            inputStream = TensorGPU.class.getClassLoader().getResourceAsStream(resourceName.substring(1));
        }

        if (inputStream == null) {
            inputStream = ClassLoader.getSystemResourceAsStream(resourceName);
        }

        if (inputStream == null) {
            inputStream = ClassLoader.getSystemResourceAsStream(resourceName.substring(1));
        }

        if (inputStream == null) {
            throw new NativeException("Resource not found: " + resourceName);
        }

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
            StringBuilder stringBuilder = new StringBuilder();
            String line;

            while ((line = reader.readLine()) != null) {
                stringBuilder.append(line).append("\n");
            }

            return stringBuilder.toString();
        } catch (Exception e) {
            throw new NativeException("Exception caught reading kernel: " + resourceName + "! " + e.getMessage());
        }
    }
 
    private static boolean checkKernelStatus(cl_program program, cl_device_id device, String kernelName) {
        try {
            int[] buildStatus = new int[1];
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, 
                    Sizeof.cl_int, Pointer.to(buildStatus), null);
                    
            if (buildStatus[0] != CL_BUILD_SUCCESS) {
                long[] logSize = new long[1];
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
                        0, null, logSize);
                        
                byte[] buildLog = new byte[(int)logSize[0]];
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
                        logSize[0], Pointer.to(buildLog), null);
                        
                System.err.println("Error in kernel compilation " + kernelName + ":");
                System.err.println(new String(buildLog));
                return false;
            }
            
            return true;
        } catch (Exception e) {
            System.err.println("Error during kernel verification " + kernelName + ": " + e.getMessage());
            return false;
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
                if (MAT_MULT_KERNEL != null) clReleaseKernel(MAT_MULT_KERNEL);
                if (ELEMENT_WISE_ADD_KERNEL != null) clReleaseKernel(ELEMENT_WISE_ADD_KERNEL);
                if (ELEMENT_WISE_MULT_KERNEL != null) clReleaseKernel(ELEMENT_WISE_MULT_KERNEL);
                
                if (CONVOLVE_1D_DIRECT_KERNEL != null) clReleaseKernel(CONVOLVE_1D_DIRECT_KERNEL);
                if (CONVOLVE_2D_DIRECT_KERNEL != null) clReleaseKernel(CONVOLVE_2D_DIRECT_KERNEL);
                
                if (FFT_1D_KERNEL != null) clReleaseKernel(FFT_1D_KERNEL);
                if (FFT_2D_KERNEL != null) clReleaseKernel(FFT_2D_KERNEL);
                if (FFT_2D_TRANSPOSE_KERNEL != null) clReleaseKernel(FFT_2D_TRANSPOSE_KERNEL);
                if (COMPLEX_POINTWISE_MUL_KERNEL != null) clReleaseKernel(COMPLEX_POINTWISE_MUL_KERNEL);
                if (CONVOLVE_1D_FFT_KERNEL != null) clReleaseKernel(CONVOLVE_1D_FFT_KERNEL);
                if (CONVOLVE_2D_FFT_EXTRACT_KERNEL != null) clReleaseKernel(CONVOLVE_2D_FFT_EXTRACT_KERNEL);
                
                if (MAIN_PROGRAM != null) clReleaseProgram(MAIN_PROGRAM);
                
                if (COMMAND_QUEUE != null) clReleaseCommandQueue(COMMAND_QUEUE);
                if (CONTEXT != null) clReleaseContext(CONTEXT);
                
                INITIALIZED = false;
            } catch (Exception e) {
                System.err.println("Error releasing GPU resources: " + e.getMessage());
            }
        }
    }

    public static void reinitializeGPU() {        
        if (!INITIALIZED) {
            try {
                INITIALIZED = initializeOpenCL();
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
            if (dimension() != 2 || other.dimension() != 2) {
                throw new IllegalArgumentException("Both tensors must be 2D for matrix multiplication");
            }
            
            int[] shapeA = shape();
            int[] shapeB = other.shape();
            
            if (shapeA[1] != shapeB[0]) {
                throw new IllegalArgumentException("Incompatible shapes for matrix multiplication: " +
                        Arrays.toString(shapeA) + " and " + Arrays.toString(shapeB));
            }
            
            int M = shapeA[0];
            int K = shapeA[1];
            int N = shapeB[1];
            
            float[] dataA = toArray();
            float[] dataB = other.toArray();
            
            cl_mem memA = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * dataA.length, Pointer.to(dataA), null);
            cl_mem memB = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * dataB.length, Pointer.to(dataB), null);
            
            float[] resultData = new float[M * N];
            cl_mem memC = clCreateBuffer(CONTEXT, CL_MEM_WRITE_ONLY,
                    (long) Sizeof.cl_float * resultData.length, null, null);
            
            clSetKernelArg(MAT_MULT_KERNEL, 0, Sizeof.cl_mem, Pointer.to(memA));
            clSetKernelArg(MAT_MULT_KERNEL, 1, Sizeof.cl_mem, Pointer.to(memB));
            clSetKernelArg(MAT_MULT_KERNEL, 2, Sizeof.cl_mem, Pointer.to(memC));
            clSetKernelArg(MAT_MULT_KERNEL, 3, Sizeof.cl_int, Pointer.to(new int[] { M }));
            clSetKernelArg(MAT_MULT_KERNEL, 4, Sizeof.cl_int, Pointer.to(new int[] { K }));
            clSetKernelArg(MAT_MULT_KERNEL, 5, Sizeof.cl_int, Pointer.to(new int[] { N }));
            
            long[] globalWorkSize = new long[] { M, N };
            clEnqueueNDRangeKernel(COMMAND_QUEUE, MAT_MULT_KERNEL, 2, null,
                globalWorkSize, null, 0, null, null);
            
            clEnqueueReadBuffer(COMMAND_QUEUE, memC, CL_TRUE, 0,
                    (long) Sizeof.cl_float * resultData.length, Pointer.to(resultData),
                    0, null, null);
            
            clReleaseMemObject(memA);
            clReleaseMemObject(memB);
            clReleaseMemObject(memC);
            
            return TensorGPU.of(new int[] { M, N }, resultData);
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
            if (!Arrays.equals(shape(), other.shape())) {
                throw new IllegalArgumentException("Tensors must have the same shape for addition");
            }
            
            int elements = elements();
            float[] dataA = toArray();
            float[] dataB = other.toArray();
            
            cl_mem memA = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * dataA.length, Pointer.to(dataA), null);
            cl_mem memB = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * dataB.length, Pointer.to(dataB), null);
            
            float[] resultData = new float[elements];
            cl_mem memC = clCreateBuffer(CONTEXT, CL_MEM_WRITE_ONLY,
                    (long) Sizeof.cl_float * resultData.length, null, null);
            
            clSetKernelArg(ELEMENT_WISE_ADD_KERNEL, 0, Sizeof.cl_mem, Pointer.to(memA));
            clSetKernelArg(ELEMENT_WISE_ADD_KERNEL, 1, Sizeof.cl_mem, Pointer.to(memB));
            clSetKernelArg(ELEMENT_WISE_ADD_KERNEL, 2, Sizeof.cl_mem, Pointer.to(memC));
            clSetKernelArg(ELEMENT_WISE_ADD_KERNEL, 3, Sizeof.cl_int, Pointer.to(new int[] { elements }));
            
            long[] globalWorkSize = new long[] { elements };
            clEnqueueNDRangeKernel(COMMAND_QUEUE, ELEMENT_WISE_ADD_KERNEL, 1, null,
                    globalWorkSize, null, 0, null, null);
            
            clEnqueueReadBuffer(COMMAND_QUEUE, memC, CL_TRUE, 0,
                    (long) Sizeof.cl_float * resultData.length, Pointer.to(resultData),
                    0, null, null);
            
            clReleaseMemObject(memA);
            clReleaseMemObject(memB);
            clReleaseMemObject(memC);
            
            return TensorGPU.of(shape(), resultData);
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
            if (!Arrays.equals(shape(), other.shape())) {
                throw new IllegalArgumentException("Tensors must have the same shape for multiplication");
            }
            
            int elements = elements();
            float[] dataA = toArray();
            float[] dataB = other.toArray();
            
            cl_mem memA = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * dataA.length, Pointer.to(dataA), null);
            cl_mem memB = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * dataB.length, Pointer.to(dataB), null);
            
            float[] resultData = new float[elements];
            cl_mem memC = clCreateBuffer(CONTEXT, CL_MEM_WRITE_ONLY,
                    (long) Sizeof.cl_float * resultData.length, null, null);
            
            clSetKernelArg(ELEMENT_WISE_MULT_KERNEL, 0, Sizeof.cl_mem, Pointer.to(memA));
            clSetKernelArg(ELEMENT_WISE_MULT_KERNEL, 1, Sizeof.cl_mem, Pointer.to(memB));
            clSetKernelArg(ELEMENT_WISE_MULT_KERNEL, 2, Sizeof.cl_mem, Pointer.to(memC));
            clSetKernelArg(ELEMENT_WISE_MULT_KERNEL, 3, Sizeof.cl_int, Pointer.to(new int[] { elements }));
            
            long[] globalWorkSize = new long[] { elements };
            clEnqueueNDRangeKernel(COMMAND_QUEUE, ELEMENT_WISE_MULT_KERNEL, 1, null,
                    globalWorkSize, null, 0, null, null);
            
            clEnqueueReadBuffer(COMMAND_QUEUE, memC, CL_TRUE, 0,
                    (long) Sizeof.cl_float * resultData.length, Pointer.to(resultData),
                    0, null, null);
            
            clReleaseMemObject(memA);
            clReleaseMemObject(memB);
            clReleaseMemObject(memC);
            
            return TensorGPU.of(shape(), resultData);
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
            if (dimension() == 1 && kernel.dimension() == 1) {
                return convolve1DGPU(kernel);
            } else if (dimension() == 2 && kernel.dimension() == 2) {
                return convolve2DGPU(kernel);
            } else {
                throw new IllegalArgumentException(
                    "Convolution supported only for 1D or 2D tensors, dimensions: " +
                    dimension() + " and " + kernel.dimension());
            }
        } catch (Exception e) {
            System.err.println("Error in GPU convolution: " + e.getMessage());
            return super.convolve(kernel);
        }
    }

    private Tensor convolve1DGPU(Tensor kernel) {
        int inputSize = shape()[0];
        int kernelSize = kernel.shape()[0];
        
        int outputSize = inputSize;  
        int totalPadding = kernelSize - 1;
        int paddingLeft = totalPadding / 2;
        
        boolean useFFT = kernelSize > FFT_THRESHOLD;
        
        float[] inputData = toArray();
        float[] kernelData = kernel.toArray();
        float[] resultData = new float[outputSize];
        
        cl_mem memInput = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                (long) Sizeof.cl_float * inputData.length, Pointer.to(inputData), null);
        cl_mem memKernel = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                (long) Sizeof.cl_float * kernelData.length, Pointer.to(kernelData), null);
        cl_mem memOutput = clCreateBuffer(CONTEXT, CL_MEM_WRITE_ONLY,
                (long) Sizeof.cl_float * resultData.length, null, null);
        
        try {
            if (!useFFT) {
                clSetKernelArg(CONVOLVE_1D_DIRECT_KERNEL, 0, Sizeof.cl_mem, Pointer.to(memInput));
                clSetKernelArg(CONVOLVE_1D_DIRECT_KERNEL, 1, Sizeof.cl_mem, Pointer.to(memKernel));
                clSetKernelArg(CONVOLVE_1D_DIRECT_KERNEL, 2, Sizeof.cl_mem, Pointer.to(memOutput));
                clSetKernelArg(CONVOLVE_1D_DIRECT_KERNEL, 3, Sizeof.cl_int, Pointer.to(new int[] { inputSize }));
                clSetKernelArg(CONVOLVE_1D_DIRECT_KERNEL, 4, Sizeof.cl_int, Pointer.to(new int[] { kernelSize }));
                clSetKernelArg(CONVOLVE_1D_DIRECT_KERNEL, 5, Sizeof.cl_int, Pointer.to(new int[] { outputSize }));
                clSetKernelArg(CONVOLVE_1D_DIRECT_KERNEL, 6, Sizeof.cl_int, Pointer.to(new int[] { paddingLeft }));
                
                long[] globalWorkSize = new long[] { outputSize };
                
                try {
                    long[] localWorkSize = getOptimalWorkgroupSize(outputSize);
                    clEnqueueNDRangeKernel(COMMAND_QUEUE, CONVOLVE_1D_DIRECT_KERNEL, 1, null,
                            globalWorkSize, localWorkSize, 0, null, null);
                } catch (Exception e) {
                    if (e.getMessage().contains("CL_INVALID_WORK_GROUP_SIZE")) { // shitty heuristics but it works
                        clEnqueueNDRangeKernel(COMMAND_QUEUE, CONVOLVE_1D_DIRECT_KERNEL, 1, null,
                                globalWorkSize, null, 0, null, null);
                    } else {
                        throw e;
                    }
                }
            } else {
                int fftSize = nextPowerOf2(inputSize + kernelSize - 1);
                
                cl_mem memBuffer1 = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                        (long) Sizeof.cl_float * 2 * fftSize, null, null);
                cl_mem memBuffer2 = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                        (long) Sizeof.cl_float * 2 * fftSize, null, null);
                cl_mem memBuffer3 = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                        (long) Sizeof.cl_float * 2 * fftSize, null, null);
                
                try {
                    clSetKernelArg(CONVOLVE_1D_FFT_KERNEL, 0, Sizeof.cl_mem, Pointer.to(memInput));
                    clSetKernelArg(CONVOLVE_1D_FFT_KERNEL, 1, Sizeof.cl_mem, Pointer.to(memKernel));
                    clSetKernelArg(CONVOLVE_1D_FFT_KERNEL, 2, Sizeof.cl_mem, Pointer.to(memOutput));
                    clSetKernelArg(CONVOLVE_1D_FFT_KERNEL, 3, Sizeof.cl_mem, Pointer.to(memBuffer1));
                    clSetKernelArg(CONVOLVE_1D_FFT_KERNEL, 4, Sizeof.cl_mem, Pointer.to(memBuffer2));
                    clSetKernelArg(CONVOLVE_1D_FFT_KERNEL, 5, Sizeof.cl_mem, Pointer.to(memBuffer3));
                    clSetKernelArg(CONVOLVE_1D_FFT_KERNEL, 6, Sizeof.cl_int, Pointer.to(new int[] { inputSize }));
                    clSetKernelArg(CONVOLVE_1D_FFT_KERNEL, 7, Sizeof.cl_int, Pointer.to(new int[] { kernelSize }));
                    clSetKernelArg(CONVOLVE_1D_FFT_KERNEL, 8, Sizeof.cl_int, Pointer.to(new int[] { fftSize }));
                    clSetKernelArg(CONVOLVE_1D_FFT_KERNEL, 9, Sizeof.cl_int, Pointer.to(new int[] { outputSize }));
                    clSetKernelArg(CONVOLVE_1D_FFT_KERNEL, 10, Sizeof.cl_int, Pointer.to(new int[] { paddingLeft }));
                    
                    long[] globalWorkSize = new long[] { fftSize };
                    
                    try {
                        long[] localWorkSize = getOptimalWorkgroupSize(fftSize);
                        clEnqueueNDRangeKernel(COMMAND_QUEUE, CONVOLVE_1D_FFT_KERNEL, 1, null,
                                globalWorkSize, localWorkSize, 0, null, null);
                    } catch (Exception e) {
                        if (e.getMessage().contains("CL_INVALID_WORK_GROUP_SIZE")) {
                            clEnqueueNDRangeKernel(COMMAND_QUEUE, CONVOLVE_1D_FFT_KERNEL, 1, null,
                                    globalWorkSize, null, 0, null, null);
                        } else {
                            throw e;
                        }
                    }
                } finally {
                    clReleaseMemObject(memBuffer1);
                    clReleaseMemObject(memBuffer2);
                    clReleaseMemObject(memBuffer3);
                }
            }
            
            clEnqueueReadBuffer(COMMAND_QUEUE, memOutput, CL_TRUE, 0,
                    (long) Sizeof.cl_float * resultData.length, Pointer.to(resultData),
                    0, null, null);
        } finally {
            clReleaseMemObject(memInput);
            clReleaseMemObject(memKernel);
            clReleaseMemObject(memOutput);
        }
        
        return TensorGPU.of(new int[] { outputSize }, resultData);
    }
    
    private Tensor convolve2DGPU(Tensor kernel) {
        int[] inputShape = shape();
        int[] kernelShape = kernel.shape();
        
        int inputRows = inputShape[0];
        int inputCols = inputShape[1];
        int kernelRows = kernelShape[0];
        int kernelCols = kernelShape[1];
        
        int outputRows = inputRows;
        int outputCols = inputCols;
        
        int paddingTop = (kernelRows - 1) / 2;
        int paddingLeft = (kernelCols - 1) / 2;
        
        boolean useFFT = (kernelRows * kernelCols) > FFT_THRESHOLD;
        
        float[] inputData = toArray();
        float[] kernelData = kernel.toArray();
        float[] resultData = new float[outputRows * outputCols];
        
        cl_mem memInput = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                (long) Sizeof.cl_float * inputData.length, Pointer.to(inputData), null);
        cl_mem memKernel = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                (long) Sizeof.cl_float * kernelData.length, Pointer.to(kernelData), null);
        cl_mem memOutput = clCreateBuffer(CONTEXT, CL_MEM_WRITE_ONLY,
                (long) Sizeof.cl_float * resultData.length, null, null);
        
        try {
            if (!useFFT) {
                clSetKernelArg(CONVOLVE_2D_DIRECT_KERNEL, 0, Sizeof.cl_mem, Pointer.to(memInput));
                clSetKernelArg(CONVOLVE_2D_DIRECT_KERNEL, 1, Sizeof.cl_mem, Pointer.to(memKernel));
                clSetKernelArg(CONVOLVE_2D_DIRECT_KERNEL, 2, Sizeof.cl_mem, Pointer.to(memOutput));
                clSetKernelArg(CONVOLVE_2D_DIRECT_KERNEL, 3, Sizeof.cl_int, Pointer.to(new int[] { inputRows }));
                clSetKernelArg(CONVOLVE_2D_DIRECT_KERNEL, 4, Sizeof.cl_int, Pointer.to(new int[] { inputCols }));
                clSetKernelArg(CONVOLVE_2D_DIRECT_KERNEL, 5, Sizeof.cl_int, Pointer.to(new int[] { kernelRows }));
                clSetKernelArg(CONVOLVE_2D_DIRECT_KERNEL, 6, Sizeof.cl_int, Pointer.to(new int[] { kernelCols }));
                clSetKernelArg(CONVOLVE_2D_DIRECT_KERNEL, 7, Sizeof.cl_int, Pointer.to(new int[] { outputRows }));
                clSetKernelArg(CONVOLVE_2D_DIRECT_KERNEL, 8, Sizeof.cl_int, Pointer.to(new int[] { outputCols }));
                clSetKernelArg(CONVOLVE_2D_DIRECT_KERNEL, 9, Sizeof.cl_int, Pointer.to(new int[] { paddingTop }));
                clSetKernelArg(CONVOLVE_2D_DIRECT_KERNEL, 10, Sizeof.cl_int, Pointer.to(new int[] { paddingLeft }));
                
                long[] globalWorkSize = new long[] { outputRows, outputCols };
                
                try {
                    long[] localWorkSize = getOptimalWorkgroupSize2D(outputRows, outputCols);
                    clEnqueueNDRangeKernel(COMMAND_QUEUE, CONVOLVE_2D_DIRECT_KERNEL, 2, null,
                            globalWorkSize, localWorkSize, 0, null, null);
                } catch (Exception e) {
                    if (e.getMessage().contains("CL_INVALID_WORK_GROUP_SIZE")) { // again :(
                        clEnqueueNDRangeKernel(COMMAND_QUEUE, CONVOLVE_2D_DIRECT_KERNEL, 2, null,
                                globalWorkSize, null, 0, null, null);
                    } else {
                        throw e;
                    }
                }
            } else {
                int fftRows = nextPowerOf2(inputRows + kernelRows - 1);
                int fftCols = nextPowerOf2(inputCols + kernelCols - 1);
                
                cl_mem memBuffer1 = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                        (long) Sizeof.cl_float * 2 * fftRows * fftCols, null, null);
                cl_mem memBuffer2 = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                        (long) Sizeof.cl_float * 2 * fftRows * fftCols, null, null);
                cl_mem memBuffer3 = clCreateBuffer(CONTEXT, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                        (long) Sizeof.cl_float * 2 * fftRows * fftCols, null, null);
                
                try {
                    clSetKernelArg(FFT_2D_KERNEL, 0, Sizeof.cl_mem, Pointer.to(memInput));
                    clSetKernelArg(FFT_2D_KERNEL, 1, Sizeof.cl_mem, Pointer.to(memBuffer1));
                    clSetKernelArg(FFT_2D_KERNEL, 2, Sizeof.cl_int, Pointer.to(new int[] { inputRows }));
                    clSetKernelArg(FFT_2D_KERNEL, 3, Sizeof.cl_int, Pointer.to(new int[] { inputCols }));
                    clSetKernelArg(FFT_2D_KERNEL, 4, Sizeof.cl_int, Pointer.to(new int[] { 1 })); // direction = 1 for forward FFT
                    
                    long[] globalWorkSize = new long[] { fftRows, fftCols };
                    
                    try {
                        long[] localWorkSize = getOptimalWorkgroupSize2D(fftRows, fftCols);
                        clEnqueueNDRangeKernel(COMMAND_QUEUE, FFT_2D_KERNEL, 2, null,
                                globalWorkSize, localWorkSize, 0, null, null);
                    } catch (Exception e) {
                        if (e.getMessage().contains("CL_INVALID_WORK_GROUP_SIZE")) { 
                            clEnqueueNDRangeKernel(COMMAND_QUEUE, FFT_2D_KERNEL, 2, null,
                                    globalWorkSize, null, 0, null, null);
                        } else {
                            throw e;
                        }
                    }
                    
                    clSetKernelArg(FFT_2D_KERNEL, 0, Sizeof.cl_mem, Pointer.to(memKernel));
                    clSetKernelArg(FFT_2D_KERNEL, 1, Sizeof.cl_mem, Pointer.to(memBuffer2));
                    clSetKernelArg(FFT_2D_KERNEL, 2, Sizeof.cl_int, Pointer.to(new int[] { kernelRows }));
                    clSetKernelArg(FFT_2D_KERNEL, 3, Sizeof.cl_int, Pointer.to(new int[] { kernelCols }));
                    clSetKernelArg(FFT_2D_KERNEL, 4, Sizeof.cl_int, Pointer.to(new int[] { 1 })); // same as above
                    
                    try {
                        long[] localWorkSize = getOptimalWorkgroupSize2D(fftRows, fftCols);
                        clEnqueueNDRangeKernel(COMMAND_QUEUE, FFT_2D_KERNEL, 2, null,
                                globalWorkSize, localWorkSize, 0, null, null);
                    } catch (Exception e) {
                        if (e.getMessage().contains("CL_INVALID_WORK_GROUP_SIZE")) {
                            clEnqueueNDRangeKernel(COMMAND_QUEUE, FFT_2D_KERNEL, 2, null,
                                    globalWorkSize, null, 0, null, null);
                        } else {
                            throw e;
                        }
                    }
                    
                    clSetKernelArg(COMPLEX_POINTWISE_MUL_KERNEL, 0, Sizeof.cl_mem, Pointer.to(memBuffer1));
                    clSetKernelArg(COMPLEX_POINTWISE_MUL_KERNEL, 1, Sizeof.cl_mem, Pointer.to(memBuffer2));
                    clSetKernelArg(COMPLEX_POINTWISE_MUL_KERNEL, 2, Sizeof.cl_mem, Pointer.to(memBuffer3));
                    clSetKernelArg(COMPLEX_POINTWISE_MUL_KERNEL, 3, Sizeof.cl_int, Pointer.to(new int[] { fftRows * fftCols }));
                    
                    try {
                        long[] localWorkSize = getOptimalWorkgroupSize((long) fftRows * fftCols);
                        clEnqueueNDRangeKernel(COMMAND_QUEUE, COMPLEX_POINTWISE_MUL_KERNEL, 1, null,
                                new long[] {(long) fftRows * fftCols }, localWorkSize, 0, null, null);
                    } catch (Exception e) {
                        if (e.getMessage().contains("CL_INVALID_WORK_GROUP_SIZE")) {
                            clEnqueueNDRangeKernel(COMMAND_QUEUE, COMPLEX_POINTWISE_MUL_KERNEL, 1, null,
                                    new long[] {(long) fftRows * fftCols }, null, 0, null, null);
                        } else {
                            throw e;
                        }
                    }
                    
                    clSetKernelArg(FFT_2D_KERNEL, 0, Sizeof.cl_mem, Pointer.to(memBuffer3));
                    clSetKernelArg(FFT_2D_KERNEL, 1, Sizeof.cl_mem, Pointer.to(memOutput));
                    clSetKernelArg(FFT_2D_KERNEL, 2, Sizeof.cl_int, Pointer.to(new int[] { fftRows }));
                    clSetKernelArg(FFT_2D_KERNEL, 3, Sizeof.cl_int, Pointer.to(new int[] { fftCols }));
                    clSetKernelArg(FFT_2D_KERNEL, 4, Sizeof.cl_int, Pointer.to(new int[] { -1 })); // direction = -1 per IFFT
                    
                    try {
                        long[] localWorkSize = getOptimalWorkgroupSize2D(fftRows, fftCols);
                        clEnqueueNDRangeKernel(COMMAND_QUEUE, FFT_2D_KERNEL, 2, null,
                                globalWorkSize, localWorkSize, 0, null, null);
                    } catch (Exception e) {
                        if (e.getMessage().contains("CL_INVALID_WORK_GROUP_SIZE")) {
                            clEnqueueNDRangeKernel(COMMAND_QUEUE, FFT_2D_KERNEL, 2, null,
                                    globalWorkSize, null, 0, null, null);
                        } else {
                            throw e;
                        }
                    }
                    
                    clSetKernelArg(CONVOLVE_2D_FFT_EXTRACT_KERNEL, 0, Sizeof.cl_mem, Pointer.to(memOutput));
                    clSetKernelArg(CONVOLVE_2D_FFT_EXTRACT_KERNEL, 1, Sizeof.cl_mem, Pointer.to(memOutput));
                    clSetKernelArg(CONVOLVE_2D_FFT_EXTRACT_KERNEL, 2, Sizeof.cl_int, Pointer.to(new int[] { outputRows }));
                    clSetKernelArg(CONVOLVE_2D_FFT_EXTRACT_KERNEL, 3, Sizeof.cl_int, Pointer.to(new int[] { outputCols }));
                    clSetKernelArg(CONVOLVE_2D_FFT_EXTRACT_KERNEL, 4, Sizeof.cl_int, Pointer.to(new int[] { paddingTop }));
                    clSetKernelArg(CONVOLVE_2D_FFT_EXTRACT_KERNEL, 5, Sizeof.cl_int, Pointer.to(new int[] { paddingLeft }));
                    clSetKernelArg(CONVOLVE_2D_FFT_EXTRACT_KERNEL, 6, Sizeof.cl_int, Pointer.to(new int[] { fftCols }));
                    
                    try {
                        long[] localWorkSize = getOptimalWorkgroupSize2D(outputRows, outputCols);
                        clEnqueueNDRangeKernel(COMMAND_QUEUE, CONVOLVE_2D_FFT_EXTRACT_KERNEL, 2, null,
                                new long[] { outputRows, outputCols }, localWorkSize, 0, null, null);
                    } catch (Exception e) {
                        if (e.getMessage().contains("CL_INVALID_WORK_GROUP_SIZE")) {
                            clEnqueueNDRangeKernel(COMMAND_QUEUE, CONVOLVE_2D_FFT_EXTRACT_KERNEL, 2, null,
                                    new long[] { outputRows, outputCols }, null, 0, null, null);
                        } else {
                            throw e;
                        }
                    }
                } finally {
                    clReleaseMemObject(memBuffer1);
                    clReleaseMemObject(memBuffer2);
                    clReleaseMemObject(memBuffer3);
                }
            }
            
            clEnqueueReadBuffer(COMMAND_QUEUE, memOutput, CL_TRUE, 0,
                    (long) Sizeof.cl_float * resultData.length, Pointer.to(resultData),
                    0, null, null);
        } finally {
            clReleaseMemObject(memInput);
            clReleaseMemObject(memKernel);
            clReleaseMemObject(memOutput);
        }
        
        return TensorGPU.of(new int[] { outputRows, outputCols }, resultData);
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
        while (size < OPTIMAL_WORKGROUP_SIZE && size * 2 <= MAX_WORKGROUP_SIZE) {
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