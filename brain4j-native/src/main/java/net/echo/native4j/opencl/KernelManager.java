package net.echo.native4j.opencl;

import org.jocl.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

import static org.jocl.CL.*;

public class KernelManager {

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
    
    private final cl_context context;
    private final cl_device_id deviceId;
    private cl_program mainProgram;
    private final Map<String, cl_kernel> kernels;
    
    public KernelManager(cl_context context, cl_device_id deviceId) {
        this.context = context;
        this.deviceId = deviceId;
        this.kernels = new HashMap<>();
    }

    public boolean initializeKernels() {
        try {
            String combinedKernelSource = loadAndCombineKernels();
            
            mainProgram = clCreateProgramWithSource(context, 1, 
                    new String[] { combinedKernelSource }, null, null);
            
            int buildResult = clBuildProgram(mainProgram, 0, null, 
                    "-cl-mad-enable -cl-fast-relaxed-math", null, null);
            
            if (buildResult != CL_SUCCESS) {
                checkKernelStatus(mainProgram, deviceId, "main_program");
                return false;
            }
            
            loadKernel("matmul");
            loadKernel("element_wise_add");
            loadKernel("element_wise_mul");
            
            loadKernel("convolve1d_direct");
            loadKernel("convolve2d_direct");
            
            loadKernel("fft1d");
            loadKernel("fft2d");
            loadKernel("fft2d_transpose");
            loadKernel("complex_pointwise_mul");
            loadKernel("convolve1d_fft");
            loadKernel("convolve2d_fft_extract");
            
            return true;
        } catch (Exception e) {
            System.err.println("Error initializing kernels: " + e.getMessage());
            return false;
        }
    }
    
    private cl_kernel loadKernel(String kernelName) {
        cl_kernel kernel = clCreateKernel(mainProgram, kernelName, null);
        kernels.put(kernelName, kernel);
        return kernel;
    }
    
    public cl_kernel getKernel(String kernelName) {
        return kernels.get(kernelName);
    }

    private String loadAndCombineKernels() throws IOException {
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
    
    private String removeIncludes(String source) {
        StringBuilder result = new StringBuilder();
        String[] lines = source.split("\n");
        
        for (String line : lines) {
            if (!line.trim().startsWith("#include")) {
                result.append(line).append("\n");
            }
        }
        
        return result.toString();
    }

    private String loadKernelSource(String resourceName) throws IOException {
        InputStream inputStream = KernelManager.class.getResourceAsStream(resourceName);
        
        if (inputStream == null) {
            inputStream = KernelManager.class.getClassLoader().getResourceAsStream(resourceName.substring(1));
        }

        if (inputStream == null) {
            inputStream = ClassLoader.getSystemResourceAsStream(resourceName);
        }

        if (inputStream == null) {
            inputStream = ClassLoader.getSystemResourceAsStream(resourceName.substring(1));
        }

        if (inputStream == null) {
            throw new IOException("Error loading kernel: " + resourceName);
        }

        try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream))) {
            StringBuilder stringBuilder = new StringBuilder();
            String line;

            while ((line = reader.readLine()) != null) {
                stringBuilder.append(line).append("\n");
            }

            return stringBuilder.toString();
        } catch (Exception e) {
            throw new IOException("Error loading kernel: " + resourceName, e);
        }
    }
    
    private boolean checkKernelStatus(cl_program program, cl_device_id device, String kernelName) {
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
    
    public void releaseResources() {
        for (cl_kernel kernel : kernels.values()) {
            if (kernel != null) {
                clReleaseKernel(kernel);
            }
        }
        
        kernels.clear();
        
        if (mainProgram != null) {
            clReleaseProgram(mainProgram);
            mainProgram = null;
        }
    }
} 