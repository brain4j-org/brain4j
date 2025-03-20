package net.echo.brain4j.utils.math.tensor;

import net.echo.brain4j.utils.opencl.DeviceUtils;
import org.jocl.*;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

import static org.jocl.CL.*;

public class TensorGPU extends Tensor {

    private static cl_kernel MAT_MULT_KERNEL;
    private static cl_kernel ELEMENT_WISE_ADD_KERNEL;
    private static cl_kernel ELEMENT_WISE_MULT_KERNEL;
    private static cl_context CONTEXT;
    private static cl_command_queue COMMAND_QUEUE;
    private static boolean INITIALIZED = false;
    
    static {
        try {
            initializeOpenCL();
        } catch (Exception e) {
            System.err.println("GPU acceleration not available: " + e.getMessage());
        }
    }
    
    private static void initializeOpenCL() {
        if (INITIALIZED) {
            throw new UnsupportedOperationException("OpenCL already initialized!");
        }
        
        try {
            CL.setExceptionsEnabled(true);
            
            cl_device_id device = DeviceUtils.findDevice(DeviceUtils.DeviceType.GPU);

            if (device == null) {
                System.out.println("No GPU device found, falling back to CPU");
                device = DeviceUtils.findDevice(DeviceUtils.DeviceType.CPU);
            }
            
            cl_device_id[] devices = {device};
            CONTEXT = clCreateContext(null, 1, devices, null, null, null);
            // TODO: Replace clCreateCommandQueue with a non-deprecated call
            // We can't because MacBook do not support it
            COMMAND_QUEUE = clCreateCommandQueue(CONTEXT, device, 0, null);
            
            String kernelSource = loadKernelSource("tensor_operations.cl");
            cl_program program = clCreateProgramWithSource(CONTEXT, 1, new String[] {kernelSource}, null, null);
            
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
            
            MAT_MULT_KERNEL = clCreateKernel(program, "matmul", null);
            ELEMENT_WISE_ADD_KERNEL = clCreateKernel(program, "element_wise_add", null);
            ELEMENT_WISE_MULT_KERNEL = clCreateKernel(program, "element_wise_mul", null);
            
            INITIALIZED = true;
            System.out.println("GPU acceleration enabled using device: " + DeviceUtils.getDeviceName());
        } catch (Exception e) {
            System.err.println("Failed to initialize OpenCL: " + e.getMessage());
            e.printStackTrace();
            INITIALIZED = false;
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
            e.printStackTrace();
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
        
        int size = numel();
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
            
            long globalWorkSize = size;
            
            clEnqueueNDRangeKernel(COMMAND_QUEUE, ELEMENT_WISE_ADD_KERNEL, 1, null,
                new long[]{globalWorkSize}, null, 0, null, null);
            
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
        
        int size = numel();
        
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
            
            long globalWorkSize = size;
            
            clEnqueueNDRangeKernel(COMMAND_QUEUE, ELEMENT_WISE_MULT_KERNEL, 1, null,
                new long[]{globalWorkSize}, null, 0, null, null);
            
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
    
    
    public static void releaseGPUResources() {
        if (INITIALIZED) {
            try {
                if (MAT_MULT_KERNEL != null) clReleaseKernel(MAT_MULT_KERNEL);
                if (ELEMENT_WISE_ADD_KERNEL != null) clReleaseKernel(ELEMENT_WISE_ADD_KERNEL);
                if (ELEMENT_WISE_MULT_KERNEL != null) clReleaseKernel(ELEMENT_WISE_MULT_KERNEL);
                if (COMMAND_QUEUE != null) clReleaseCommandQueue(COMMAND_QUEUE);
                if (CONTEXT != null) clReleaseContext(CONTEXT);
                
                MAT_MULT_KERNEL = null;
                ELEMENT_WISE_ADD_KERNEL = null;
                ELEMENT_WISE_MULT_KERNEL = null;
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
} 