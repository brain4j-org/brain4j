package net.echo.math4j.math.tensor.impl;

import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.opencl.DeviceUtils;
import org.jocl.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Arrays;

import static org.jocl.CL.*;

public class TensorGPU extends TensorCPU {

    private static boolean INITIALIZED;
    private static cl_device_id DEVICE_ID;
    private static cl_context CONTEXT;
    private static cl_command_queue COMMAND_QUEUE;
    private static cl_program TENSOR_PROGRAM;
    private static cl_program CONV_PROGRAM;
    private static cl_kernel MAT_MULT_KERNEL;
    private static cl_kernel ELEMENT_WISE_ADD_KERNEL;
    private static cl_kernel ELEMENT_WISE_MULT_KERNEL;
    private static cl_kernel CONVOLVE_1D_KERNEL;
    private static cl_kernel CONVOLVE_2D_KERNEL;
    
    private static final String TENSOR_OPS_KERNEL_PATH = "/kernels/tensor_operations.cl";
    private static final String CONV_KERNEL_PATH = "/kernels/conv.cl";
    
    static {
        try {
            INITIALIZED = initializeOpenCL();
        } catch (Exception e) {
            System.err.println("GPU acceleration not available: " + e.getMessage());
        }
    }
    
    private static boolean initializeOpenCL() {
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
                String kernelSourceTensor = loadKernelSource(TENSOR_OPS_KERNEL_PATH);
                TENSOR_PROGRAM = clCreateProgramWithSource(CONTEXT, 1, 
                        new String[] { kernelSourceTensor }, null, null);
                int buildResult = clBuildProgram(TENSOR_PROGRAM, 0, null, null, null, null);
                
                if (buildResult != CL_SUCCESS) {
                    checkKernelStatus(TENSOR_PROGRAM, DEVICE_ID, "tensor_operations");
                    return false;
                }
                
                MAT_MULT_KERNEL = clCreateKernel(TENSOR_PROGRAM, "matmul", null);
                ELEMENT_WISE_ADD_KERNEL = clCreateKernel(TENSOR_PROGRAM, "element_wise_add", null);
                ELEMENT_WISE_MULT_KERNEL = clCreateKernel(TENSOR_PROGRAM, "element_wise_mul", null);
                
                String convKernelSource = loadKernelSource(CONV_KERNEL_PATH);
                CONV_PROGRAM = clCreateProgramWithSource(CONTEXT, 1, 
                        new String[] { convKernelSource }, null, null);
                buildResult = clBuildProgram(CONV_PROGRAM, 0, null, null, null, null);
                
                if (buildResult != CL_SUCCESS) {
                    checkKernelStatus(CONV_PROGRAM, DEVICE_ID, "convolution");
                    return false;
                }
                
                CONVOLVE_1D_KERNEL = clCreateKernel(CONV_PROGRAM, "convolve1d", null);
                CONVOLVE_2D_KERNEL = clCreateKernel(CONV_PROGRAM, "convolve2d", null);
                
                System.out.println("GPU acceleration enabled using device: " + DeviceUtils.getDeviceName());
                return true;
            } catch (Exception e) {
                System.err.println("Error loading or compiling kernels: " + e.getMessage());
                return false;
            }
        } catch (Exception e) {
            System.err.println("Failed to initialize OpenCL: " + e.getMessage());
            return false;
        }
    }
    
    private static String loadKernelSource(String resourceName) throws IOException {
        InputStream is = TensorGPU.class.getResourceAsStream(resourceName);
        
        if (is == null) {
            is = TensorGPU.class.getClassLoader().getResourceAsStream(resourceName.substring(1));
            
            if (is == null) {
                is = ClassLoader.getSystemResourceAsStream(resourceName);
                
                if (is == null) {
                    is = ClassLoader.getSystemResourceAsStream(resourceName.substring(1));
                    
                    if (is == null) {
                        throw new IOException("Error loading kernel: " + resourceName);
                    }
                }
            }
        }
        
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {
            StringBuilder sb = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append("\n");
            }
            return sb.toString();
        } catch (Exception e) {
            throw new IOException("Error loading kernel: " + resourceName, e);
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
                if (CONVOLVE_1D_KERNEL != null) clReleaseKernel(CONVOLVE_1D_KERNEL);
                if (CONVOLVE_2D_KERNEL != null) clReleaseKernel(CONVOLVE_2D_KERNEL);
                
                if (TENSOR_PROGRAM != null) clReleaseProgram(TENSOR_PROGRAM);
                if (CONV_PROGRAM != null) clReleaseProgram(CONV_PROGRAM);
                
                if (COMMAND_QUEUE != null) clReleaseCommandQueue(COMMAND_QUEUE);
                if (CONTEXT != null) clReleaseContext(CONTEXT);
                
                MAT_MULT_KERNEL = null;
                ELEMENT_WISE_ADD_KERNEL = null;
                ELEMENT_WISE_MULT_KERNEL = null;
                CONVOLVE_1D_KERNEL = null;
                CONVOLVE_2D_KERNEL = null;
                TENSOR_PROGRAM = null;
                CONV_PROGRAM = null;
                COMMAND_QUEUE = null;
                CONTEXT = null;
                
                INITIALIZED = false;
                System.out.println("GPU resources released successfully");
            } catch (Exception e) {
                System.err.println("Error releasing GPU resources: " + e.getMessage());
            }
        }
    }

    public static void reinitializeGPU() {
        releaseGPUResources();

        try {
            initializeOpenCL();
        } catch (Exception e) {
            System.err.println("Failed to reinitialize GPU: " + e.getMessage());
        }
    }

    private static int[] linearToMultiDimIndices(int linearIndex, int[] shape) {
        int[] indices = new int[shape.length];

        for (int i = shape.length - 1; i >= 0; i--) {
            indices[i] = linearIndex % shape[i];
            linearIndex /= shape[i];
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
        
        int[] thisShape = shape();
        int[] otherShape = other.shape();
        
        if (thisShape.length != 2 || otherShape.length != 2) {
            throw new IllegalArgumentException("matmul requires 2D tensors");
        }
        
        int m = thisShape[0];         
        int n = thisShape[1];         
        int p = otherShape[1]; 
        
        if (n != otherShape[0]) {
            throw new IllegalArgumentException("The inner dimensions do not match: " + n + " != " + otherShape[0]);
        }
        
        TensorGPU result = new TensorGPU(m, p);
        
        try {
            cl_mem deviceA = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * m * n, Pointer.to(this.toArray()), null);
            cl_mem deviceB = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * n * p, Pointer.to(other.toArray()), null);
            cl_mem deviceC = clCreateBuffer(CONTEXT, CL_MEM_WRITE_ONLY,
                    (long) Sizeof.cl_float * m * p, null, null);
            
            clSetKernelArg(MAT_MULT_KERNEL, 0, Sizeof.cl_mem, Pointer.to(deviceA));
            clSetKernelArg(MAT_MULT_KERNEL, 1, Sizeof.cl_mem, Pointer.to(deviceB));
            clSetKernelArg(MAT_MULT_KERNEL, 2, Sizeof.cl_mem, Pointer.to(deviceC));
            clSetKernelArg(MAT_MULT_KERNEL, 3, Sizeof.cl_int, Pointer.to(new int[]{m}));
            clSetKernelArg(MAT_MULT_KERNEL, 4, Sizeof.cl_int, Pointer.to(new int[]{n}));
            clSetKernelArg(MAT_MULT_KERNEL, 5, Sizeof.cl_int, Pointer.to(new int[]{p}));
            
            long[] globalWorkSize = new long[]{m, p};
            
            clEnqueueNDRangeKernel(COMMAND_QUEUE, MAT_MULT_KERNEL, 2, null,
                globalWorkSize, null, 0, null, null);
            
            float[] resultBuffer = new float[m * p];
            clEnqueueReadBuffer(COMMAND_QUEUE, deviceC, CL_TRUE, 0,
                    (long) Sizeof.cl_float * m * p, Pointer.to(resultBuffer), 0, null, null);
            
            for (int i = 0; i < m * p; i++) {
                int row = i / p;
                int col = i % p;
                result.set(resultBuffer[i], row, col);
            }
            
            clReleaseMemObject(deviceA);
            clReleaseMemObject(deviceB);
            clReleaseMemObject(deviceC);
            
        } catch (Exception e) {
            System.err.println("GPU matrix multiplication failed: " + e.getMessage());
            return super.matmul(other);
        }
        
        return result;
    }
    
    @Override
    public Tensor add(Tensor other) {
        int[] thisShape = shape();
        int[] otherShape = other.shape();
        
        if (!INITIALIZED || !Arrays.equals(thisShape, otherShape)) {
            return super.add(other); 
        }
        
        int size = elements();
        long floatSize = (long) Sizeof.cl_float * size;
        
        try {
            cl_mem deviceA = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    floatSize, Pointer.to(this.toArray()), null);
            cl_mem deviceB = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    floatSize, Pointer.to(other.toArray()), null);
            cl_mem deviceC = clCreateBuffer(CONTEXT, CL_MEM_WRITE_ONLY, floatSize, null, null);
            
            clSetKernelArg(ELEMENT_WISE_ADD_KERNEL, 0, Sizeof.cl_mem, Pointer.to(deviceA));
            clSetKernelArg(ELEMENT_WISE_ADD_KERNEL, 1, Sizeof.cl_mem, Pointer.to(deviceB));
            clSetKernelArg(ELEMENT_WISE_ADD_KERNEL, 2, Sizeof.cl_mem, Pointer.to(deviceC));
            clSetKernelArg(ELEMENT_WISE_ADD_KERNEL, 3, Sizeof.cl_int, Pointer.to(new int[]{size}));

            clEnqueueNDRangeKernel(COMMAND_QUEUE, ELEMENT_WISE_ADD_KERNEL, 1, null,
                new long[]{(long) size}, null, 0, null, null);
            
            float[] resultBuffer = new float[size];
            clEnqueueReadBuffer(COMMAND_QUEUE, deviceC, CL_TRUE, 0,
                    (long) Sizeof.cl_float * size, Pointer.to(resultBuffer), 0, null, null);
            
            for (int i = 0; i < size; i++) {
                if (thisShape.length == 1) {
                    this.set(resultBuffer[i], i);
                } else {
                    int[] indices = linearToMultiDimIndices(i, thisShape);
                    this.set(resultBuffer[i], indices);
                }
            }
            
            clReleaseMemObject(deviceA);
            clReleaseMemObject(deviceB);
            clReleaseMemObject(deviceC);
            
        } catch (Exception e) {
            System.err.println("GPU element-wise addition failed: " + e.getMessage());
            return super.add(other);
        }
        
        return this;
    }
    
    @Override
    public Tensor mul(Tensor other) {
        int[] thisShape = shape();
        int[] otherShape = other.shape();
        
        if (!INITIALIZED || !Arrays.equals(thisShape, otherShape)) {
            return super.mul(other); 
        }
        
        int size = elements();
        
        try {
            cl_mem deviceA = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * size, Pointer.to(this.toArray()), null);
            cl_mem deviceB = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * size, Pointer.to(other.toArray()), null);
            cl_mem deviceC = clCreateBuffer(CONTEXT, CL_MEM_WRITE_ONLY,
                    (long) Sizeof.cl_float * size, null, null);
            
            clSetKernelArg(ELEMENT_WISE_MULT_KERNEL, 0, Sizeof.cl_mem, Pointer.to(deviceA));
            clSetKernelArg(ELEMENT_WISE_MULT_KERNEL, 1, Sizeof.cl_mem, Pointer.to(deviceB));
            clSetKernelArg(ELEMENT_WISE_MULT_KERNEL, 2, Sizeof.cl_mem, Pointer.to(deviceC));
            clSetKernelArg(ELEMENT_WISE_MULT_KERNEL, 3, Sizeof.cl_int, Pointer.to(new int[]{size}));

            clEnqueueNDRangeKernel(COMMAND_QUEUE, ELEMENT_WISE_MULT_KERNEL, 1, null,
                new long[]{(long) size}, null, 0, null, null);
            
            float[] resultBuffer = new float[size];
            clEnqueueReadBuffer(COMMAND_QUEUE, deviceC, CL_TRUE, 0,
                    (long) Sizeof.cl_float * size, Pointer.to(resultBuffer), 0, null, null);
            
            for (int i = 0; i < size; i++) {
                if (thisShape.length == 1) {
                    this.set(resultBuffer[i], i);
                } else {
                    int[] indices = linearToMultiDimIndices(i, thisShape);
                    this.set(resultBuffer[i], indices);
                }
            }
            
            clReleaseMemObject(deviceA);
            clReleaseMemObject(deviceB);
            clReleaseMemObject(deviceC);
            
        } catch (Exception e) {
            System.err.println("GPU element-wise multiplication failed: " + e.getMessage());
            return super.mul(other);
        }
        
        return this;
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Tensor convolve(Tensor kernel) {
        int dim = this.dimension();
        
        if (dim > 2) {
            throw new IllegalArgumentException("Convolution is supported only for 1D and 2D tensors");
        }
        if (kernel.dimension() != dim) {
            throw new IllegalArgumentException("The kernel dimension must match the input dimension");
        }
        
        if (!INITIALIZED) {
            return super.convolve(kernel);
        }
        
        try {
            if (dim == 1) {
                return convolve1DGPU(kernel);
            } else {
                return convolve2DGPU(kernel);
            }
        } catch (Exception e) {
            System.err.println("Error in GPU convolution: " + e.getMessage());
            return super.convolve(kernel);
        }
    }

    private Tensor convolve1DGPU(Tensor kernel) {
        int inputSize = this.shape()[0];
        int kernelSize = kernel.shape()[0];
        int outputSize = inputSize;  
        int padding = kernelSize / 2;
        int stride = 1;
        
        float[] result = new float[outputSize];
        Tensor resultTensor = new TensorCPU(new int[] {outputSize});
        
        try {
            cl_mem inputBuffer = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                    Sizeof.cl_float * inputSize, Pointer.to(this.toArray()), null);
            cl_mem kernelBuffer = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                    Sizeof.cl_float * kernelSize, Pointer.to(kernel.toArray()), null);
            cl_mem outputBuffer = clCreateBuffer(CONTEXT, CL_MEM_WRITE_ONLY, 
                    Sizeof.cl_float * outputSize, null, null);
            
            int argIndex = 0;
            clSetKernelArg(CONVOLVE_1D_KERNEL, argIndex++, Sizeof.cl_mem, Pointer.to(inputBuffer));
            clSetKernelArg(CONVOLVE_1D_KERNEL, argIndex++, Sizeof.cl_mem, Pointer.to(kernelBuffer));
            clSetKernelArg(CONVOLVE_1D_KERNEL, argIndex++, Sizeof.cl_mem, Pointer.to(outputBuffer));
            clSetKernelArg(CONVOLVE_1D_KERNEL, argIndex++, Sizeof.cl_int, Pointer.to(new int[] {inputSize}));
            clSetKernelArg(CONVOLVE_1D_KERNEL, argIndex++, Sizeof.cl_int, Pointer.to(new int[] {kernelSize}));
            clSetKernelArg(CONVOLVE_1D_KERNEL, argIndex++, Sizeof.cl_int, Pointer.to(new int[] {outputSize}));
            clSetKernelArg(CONVOLVE_1D_KERNEL, argIndex++, Sizeof.cl_int, Pointer.to(new int[] {stride}));
            clSetKernelArg(CONVOLVE_1D_KERNEL, argIndex++, Sizeof.cl_int, Pointer.to(new int[] {padding}));
            
            long[] globalWorkSize = new long[] {outputSize};
            clEnqueueNDRangeKernel(COMMAND_QUEUE, CONVOLVE_1D_KERNEL, 1, null, globalWorkSize, null, 0, null, null);
            
            clEnqueueReadBuffer(COMMAND_QUEUE, outputBuffer, CL_TRUE, 0, 
                    Sizeof.cl_float * outputSize, Pointer.to(result), 0, null, null);
            
            for (int i = 0; i < outputSize; i++) {
                resultTensor.set(result[i], i);
            }
            
            clReleaseMemObject(inputBuffer);
            clReleaseMemObject(kernelBuffer);
            clReleaseMemObject(outputBuffer);
            
            return resultTensor;
        } catch (Exception e) {
            throw e;
        }
    }
    
    private Tensor convolve2DGPU(Tensor kernel) {
        int[] inputShape = this.shape();
        int[] kernelShape = kernel.shape();
        
        int inputRows = inputShape[0];
        int inputCols = inputShape[1];
        int kernelRows = kernelShape[0];
        int kernelCols = kernelShape[1];
        
        int outputRows = inputRows;
        int outputCols = inputCols;
        
        int paddingRows = kernelRows / 2;
        int paddingCols = kernelCols / 2;
        int stride = 1;
        
        float[] result = new float[outputRows * outputCols];
        Tensor resultTensor = new TensorCPU(new int[] {outputRows, outputCols});
        
        try {
            cl_mem inputBuffer = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                    Sizeof.cl_float * inputRows * inputCols, Pointer.to(this.toArray()), null);
            cl_mem kernelBuffer = clCreateBuffer(CONTEXT, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                    Sizeof.cl_float * kernelRows * kernelCols, Pointer.to(kernel.toArray()), null);
            cl_mem outputBuffer = clCreateBuffer(CONTEXT, CL_MEM_WRITE_ONLY, 
                    Sizeof.cl_float * outputRows * outputCols, null, null);
            
            int argIndex = 0;
            clSetKernelArg(CONVOLVE_2D_KERNEL, argIndex++, Sizeof.cl_mem, Pointer.to(inputBuffer));
            clSetKernelArg(CONVOLVE_2D_KERNEL, argIndex++, Sizeof.cl_mem, Pointer.to(kernelBuffer));
            clSetKernelArg(CONVOLVE_2D_KERNEL, argIndex++, Sizeof.cl_mem, Pointer.to(outputBuffer));
            clSetKernelArg(CONVOLVE_2D_KERNEL, argIndex++, Sizeof.cl_int, Pointer.to(new int[] {inputRows}));
            clSetKernelArg(CONVOLVE_2D_KERNEL, argIndex++, Sizeof.cl_int, Pointer.to(new int[] {inputCols}));
            clSetKernelArg(CONVOLVE_2D_KERNEL, argIndex++, Sizeof.cl_int, Pointer.to(new int[] {kernelRows}));
            clSetKernelArg(CONVOLVE_2D_KERNEL, argIndex++, Sizeof.cl_int, Pointer.to(new int[] {kernelCols}));
            clSetKernelArg(CONVOLVE_2D_KERNEL, argIndex++, Sizeof.cl_int, Pointer.to(new int[] {outputRows}));
            clSetKernelArg(CONVOLVE_2D_KERNEL, argIndex++, Sizeof.cl_int, Pointer.to(new int[] {outputCols}));
            clSetKernelArg(CONVOLVE_2D_KERNEL, argIndex++, Sizeof.cl_int, Pointer.to(new int[] {stride}));
            clSetKernelArg(CONVOLVE_2D_KERNEL, argIndex++, Sizeof.cl_int, Pointer.to(new int[] {paddingRows}));
            clSetKernelArg(CONVOLVE_2D_KERNEL, argIndex++, Sizeof.cl_int, Pointer.to(new int[] {paddingCols}));
            
            long[] globalWorkSize = new long[] {outputRows, outputCols};
            clEnqueueNDRangeKernel(COMMAND_QUEUE, CONVOLVE_2D_KERNEL, 2, null, globalWorkSize, null, 0, null, null);
            
            clEnqueueReadBuffer(COMMAND_QUEUE, outputBuffer, CL_TRUE, 0, 
                    Sizeof.cl_float * outputRows * outputCols, Pointer.to(result), 0, null, null);
            
            for (int i = 0; i < outputRows; i++) {
                for (int j = 0; j < outputCols; j++) {
                    int index = i * outputCols + j;
                    resultTensor.set(result[index], i, j);
                }
            }
            
            clReleaseMemObject(inputBuffer);
            clReleaseMemObject(kernelBuffer);
            clReleaseMemObject(outputBuffer);
            
            return resultTensor;
        } catch (Exception e) {
            throw e;
        }
    }
} 