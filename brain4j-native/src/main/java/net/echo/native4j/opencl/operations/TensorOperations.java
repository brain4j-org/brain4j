package net.echo.native4j.opencl.operations;

import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.impl.TensorGPU;
import net.echo.native4j.opencl.OpenCLContext;
import org.jocl.*;

import java.util.Arrays;

import static org.jocl.CL.*;

public class TensorOperations {

    public static Tensor elementWiseAdd(Tensor a, Tensor b) {
        OpenCLContext context = OpenCLContext.getInstance();
        
        if (!context.isInitialized()) {
            throw new RuntimeException("OpenCL not initialized");
        }
        
        if (!Arrays.equals(a.shape(), b.shape())) {
            throw new IllegalArgumentException("Tensors must have the same shape for addition");
        }
        
        int elements = a.elements();
        float[] dataA = a.toArray();
        float[] dataB = b.toArray();
        
        cl_mem memA = null;
        cl_mem memB = null;
        cl_mem memC = null;
        
        try {
            memA = clCreateBuffer(context.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * dataA.length, Pointer.to(dataA), null);
            memB = clCreateBuffer(context.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * dataB.length, Pointer.to(dataB), null);
            
            float[] resultData = new float[elements];
            memC = clCreateBuffer(context.getContext(), CL_MEM_WRITE_ONLY,
                    (long) Sizeof.cl_float * resultData.length, null, null);
            
            cl_kernel kernel = context.getKernel("element_wise_add");
            
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(memA));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(memB));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(memC));
            clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] { elements }));
            
            long[] globalWorkSize = new long[] { elements };
            clEnqueueNDRangeKernel(context.getCommandQueue(), kernel, 1, null,
                    globalWorkSize, null, 0, null, null);
            
            clEnqueueReadBuffer(context.getCommandQueue(), memC, CL_TRUE, 0,
                    (long) Sizeof.cl_float * resultData.length, Pointer.to(resultData),
                    0, null, null);
            
            return TensorGPU.of(a.shape(), resultData);
        } finally {
            if (memA != null) clReleaseMemObject(memA);
            if (memB != null) clReleaseMemObject(memB);
            if (memC != null) clReleaseMemObject(memC);
        }
    }

    public static Tensor elementWiseMul(Tensor a, Tensor b) {
        OpenCLContext context = OpenCLContext.getInstance();
        
        if (!context.isInitialized()) {
            throw new RuntimeException("OpenCL not initialized");
        }
        
        if (!Arrays.equals(a.shape(), b.shape())) {
            throw new IllegalArgumentException("Tensors must have the same shape for multiplication");
        }
        
        int elements = a.elements();
        float[] dataA = a.toArray();
        float[] dataB = b.toArray();
        
        cl_mem memA = null;
        cl_mem memB = null;
        cl_mem memC = null;
        
        try {
            memA = clCreateBuffer(context.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * dataA.length, Pointer.to(dataA), null);
            memB = clCreateBuffer(context.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    (long) Sizeof.cl_float * dataB.length, Pointer.to(dataB), null);
            
            float[] resultData = new float[elements];
            memC = clCreateBuffer(context.getContext(), CL_MEM_WRITE_ONLY,
                    (long) Sizeof.cl_float * resultData.length, null, null);
            
            cl_kernel kernel = context.getKernel("element_wise_mul");
            
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(memA));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(memB));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(memC));
            clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] { elements }));
            
            long[] globalWorkSize = new long[] { elements };
            clEnqueueNDRangeKernel(context.getCommandQueue(), kernel, 1, null,
                    globalWorkSize, null, 0, null, null);
            
            clEnqueueReadBuffer(context.getCommandQueue(), memC, CL_TRUE, 0,
                    (long) Sizeof.cl_float * resultData.length, Pointer.to(resultData),
                    0, null, null);
            
            return TensorGPU.of(a.shape(), resultData);
        } finally {
            if (memA != null) clReleaseMemObject(memA);
            if (memB != null) clReleaseMemObject(memB);
            if (memC != null) clReleaseMemObject(memC);
        }
    }
} 