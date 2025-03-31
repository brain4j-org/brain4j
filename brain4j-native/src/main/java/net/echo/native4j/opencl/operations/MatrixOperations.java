package net.echo.native4j.opencl.operations;

import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.impl.TensorGPU;
import net.echo.native4j.opencl.OpenCLContext;
import org.jocl.*;

import java.util.Arrays;

import static org.jocl.CL.*;

public class MatrixOperations {

    public static Tensor matmul(Tensor a, Tensor b) {
        OpenCLContext context = OpenCLContext.getInstance();
        
        if (!context.isInitialized()) {
            throw new RuntimeException("OpenCL not initialized");
        }
        
        if (a.dimension() != 2 || b.dimension() != 2) {
            throw new IllegalArgumentException("Both tensors must be 2D for matrix multiplication");
        }
        
        int[] shapeA = a.shape();
        int[] shapeB = b.shape();
        
        if (shapeA[1] != shapeB[0]) {
            throw new IllegalArgumentException("Incompatible shapes for matrix multiplication: " +
                    Arrays.toString(shapeA) + " and " + Arrays.toString(shapeB));
        }
        
        int M = shapeA[0];
        int K = shapeA[1];
        int N = shapeB[1];
        
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
            
            float[] resultData = new float[M * N];
            memC = clCreateBuffer(context.getContext(), CL_MEM_WRITE_ONLY,
                    (long) Sizeof.cl_float * resultData.length, null, null);
            
            cl_kernel kernel = context.getKernel("matmul");
            
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(memA));
            clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(memB));
            clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(memC));
            clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[] { M }));
            clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[] { K }));
            clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(new int[] { N }));
            
            long[] globalWorkSize = new long[] { M, N };
            clEnqueueNDRangeKernel(context.getCommandQueue(), kernel, 2, null,
                globalWorkSize, null, 0, null, null);
            
            clEnqueueReadBuffer(context.getCommandQueue(), memC, CL_TRUE, 0,
                    (long) Sizeof.cl_float * resultData.length, Pointer.to(resultData),
                    0, null, null);
            
            return TensorGPU.of(new int[] { M, N }, resultData);
        } finally {
            if (memA != null) clReleaseMemObject(memA);
            if (memB != null) clReleaseMemObject(memB);
            if (memC != null) clReleaseMemObject(memC);
        }
    }
} 