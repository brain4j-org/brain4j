package net.echo.brain4j.utils.math.tensor;

import net.echo.brain4j.utils.opencl.DeviceUtils;
import org.jocl.*;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

import static org.jocl.CL.*;

public class TensorGPU extends Tensor {
    
    private static cl_context context;
    private static cl_command_queue commandQueue;
    private static cl_kernel matmulKernel;
    private static cl_kernel elementWiseAddKernel;
    private static cl_kernel elementWiseMulKernel;
    private static boolean initialized = false;
    
    static {
        try {
            initializeOpenCL();
        } catch (Exception e) {
            System.err.println("GPU acceleration not available: " + e.getMessage());
        }
    }
    
    private static void initializeOpenCL() {
        if (initialized) return;
        
        try {
            CL.setExceptionsEnabled(true);
            
            cl_device_id device = DeviceUtils.findDevice(DeviceUtils.DeviceType.GPU);
            if (device == null) {
                System.out.println("No GPU device found, falling back to CPU");
                device = DeviceUtils.findDevice(DeviceUtils.DeviceType.CPU);
            }
            
            cl_device_id[] devices = {device};
            context = clCreateContext(null, 1, devices, null, null, null);
            // TODO: Replace clCreateCommandQueue with a non-deprecated call
            commandQueue = clCreateCommandQueue(context, device, 0, null);
            
            String kernelSource = loadKernelSource("tensor_operations.cl");
            cl_program program = clCreateProgramWithSource(context, 1, new String[] {kernelSource}, null, null);
            
            try {
                clBuildProgram(program, 0, null, null, null, null);
            } catch (CLException e) {
                long[] logSize = new long[1];
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, null, logSize);
                byte[] logData = new byte[(int)logSize[0]];
                clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logData.length, Pointer.to(logData), null);
                System.err.println("Build log: " + new String(logData));
                throw e;
            }
            
            matmulKernel = clCreateKernel(program, "matmul", null);
            elementWiseAddKernel = clCreateKernel(program, "element_wise_add", null);
            elementWiseMulKernel = clCreateKernel(program, "element_wise_mul", null);
            
            initialized = true;
            System.out.println("GPU acceleration enabled using device: " + DeviceUtils.getDeviceName());
        } catch (Exception e) {
            System.err.println("Failed to initialize OpenCL: " + e.getMessage());
            e.printStackTrace();
            initialized = false;
        }
    }
    
    private static String loadKernelSource(String filename) throws IOException {
        try (InputStream is = TensorGPU.class.getClassLoader().getResourceAsStream("kernels/" + filename)) {
            if (is == null) {
                throw new IOException("Kernel file not found: " + filename);
            }
            return new String(is.readAllBytes());
        }
    }
    
    public TensorGPU(int... shape) {
        super(shape);
    }
    
    public static TensorGPU fromTensor(Tensor tensor) {
        TensorGPU gpuTensor = new TensorGPU(tensor.shape());
        int[] shape = tensor.shape();
        
        for (int i = 0; i < tensor.numel(); i++) {
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
    
    public static TensorGPU of(int[] shape, float... data) {
        TensorGPU tensor = new TensorGPU(shape);
        for (int i = 0; i < data.length; i++) {
            tensor.set(data[i], i);
        }
        return tensor;
    }
    
    public static TensorGPU matrix(int rows, int cols, float... data) {
        return of(new int[]{rows, cols}, data);
    }
    
    
    public static boolean isGpuAvailable() {
        return initialized;
    }
    
    @Override
    public Tensor matmul(Tensor other) {
        if (!initialized) {
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
            throw new IllegalArgumentException(
                "The inner dimensions do not match: " + n + " != " + otherShape[0]
            );
        }
        
        TensorGPU result = new TensorGPU(m, p);
        
        try {
            cl_mem deviceA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                Sizeof.cl_float * m * n, Pointer.to(this.toArray()), null);
            cl_mem deviceB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                Sizeof.cl_float * n * p, Pointer.to(other.toArray()), null);
            cl_mem deviceC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                Sizeof.cl_float * m * p, null, null);
            
            clSetKernelArg(matmulKernel, 0, Sizeof.cl_mem, Pointer.to(deviceA));
            clSetKernelArg(matmulKernel, 1, Sizeof.cl_mem, Pointer.to(deviceB));
            clSetKernelArg(matmulKernel, 2, Sizeof.cl_mem, Pointer.to(deviceC));
            clSetKernelArg(matmulKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{m}));
            clSetKernelArg(matmulKernel, 4, Sizeof.cl_int, Pointer.to(new int[]{n}));
            clSetKernelArg(matmulKernel, 5, Sizeof.cl_int, Pointer.to(new int[]{p}));
            
            long[] globalWorkSize = new long[]{m, p};
            
            clEnqueueNDRangeKernel(commandQueue, matmulKernel, 2, null, 
                globalWorkSize, null, 0, null, null);
            
            float[] resultBuffer = new float[m * p];
            clEnqueueReadBuffer(commandQueue, deviceC, CL_TRUE, 0, 
                Sizeof.cl_float * m * p, Pointer.to(resultBuffer), 0, null, null);
            
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
            e.printStackTrace();
            return super.matmul(other);
        }
        
        return result;
    }
    
    @Override
    public Tensor add(Tensor other) {
        int[] thisShape = shape();
        int[] otherShape = other.shape();
        
        if (!initialized || !Arrays.equals(thisShape, otherShape)) {
            return super.add(other); 
        }
        
        int size = numel();
        
        try {
            cl_mem deviceA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                Sizeof.cl_float * size, Pointer.to(this.toArray()), null);
            cl_mem deviceB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                Sizeof.cl_float * size, Pointer.to(other.toArray()), null);
            cl_mem deviceC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                Sizeof.cl_float * size, null, null);
            
            clSetKernelArg(elementWiseAddKernel, 0, Sizeof.cl_mem, Pointer.to(deviceA));
            clSetKernelArg(elementWiseAddKernel, 1, Sizeof.cl_mem, Pointer.to(deviceB));
            clSetKernelArg(elementWiseAddKernel, 2, Sizeof.cl_mem, Pointer.to(deviceC));
            clSetKernelArg(elementWiseAddKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{size}));
            
            long globalWorkSize = size;
            
            clEnqueueNDRangeKernel(commandQueue, elementWiseAddKernel, 1, null, 
                new long[]{globalWorkSize}, null, 0, null, null);
            
            float[] resultBuffer = new float[size];
            clEnqueueReadBuffer(commandQueue, deviceC, CL_TRUE, 0, 
                Sizeof.cl_float * size, Pointer.to(resultBuffer), 0, null, null);
            
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
        
        if (!initialized || !Arrays.equals(thisShape, otherShape)) {
            return super.mul(other); 
        }
        
        int size = numel();
        
        try {
            cl_mem deviceA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                Sizeof.cl_float * size, Pointer.to(this.toArray()), null);
            cl_mem deviceB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                Sizeof.cl_float * size, Pointer.to(other.toArray()), null);
            cl_mem deviceC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                Sizeof.cl_float * size, null, null);
            
            clSetKernelArg(elementWiseMulKernel, 0, Sizeof.cl_mem, Pointer.to(deviceA));
            clSetKernelArg(elementWiseMulKernel, 1, Sizeof.cl_mem, Pointer.to(deviceB));
            clSetKernelArg(elementWiseMulKernel, 2, Sizeof.cl_mem, Pointer.to(deviceC));
            clSetKernelArg(elementWiseMulKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{size}));
            
            long globalWorkSize = size;
            
            clEnqueueNDRangeKernel(commandQueue, elementWiseMulKernel, 1, null, 
                new long[]{globalWorkSize}, null, 0, null, null);
            
            float[] resultBuffer = new float[size];
            clEnqueueReadBuffer(commandQueue, deviceC, CL_TRUE, 0, 
                Sizeof.cl_float * size, Pointer.to(resultBuffer), 0, null, null);
            
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
    
    
    public static void releaseGPUResources() {
        if (initialized) {
            try {
                if (matmulKernel != null) clReleaseKernel(matmulKernel);
                if (elementWiseAddKernel != null) clReleaseKernel(elementWiseAddKernel);
                if (elementWiseMulKernel != null) clReleaseKernel(elementWiseMulKernel);
                if (commandQueue != null) clReleaseCommandQueue(commandQueue);
                if (context != null) clReleaseContext(context);
                
                matmulKernel = null;
                elementWiseAddKernel = null;
                elementWiseMulKernel = null;
                commandQueue = null;
                context = null;
                initialized = false;
                
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
} 