package tensor;

import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;
import net.echo.math4j.math.tensor.impl.TensorGPU;
import net.echo.math4j.opencl.GPUProfiler;

public class TensorGPUTest {

    public static void main(String[] args) {
        System.out.println("GPU acceleration available: " + TensorGPU.isGpuAvailable());
        System.out.println("Using GPU: " + TensorFactory.isUsingGPU());
        
        if (TensorGPU.isGpuAvailable()) {
            GPUProfiler.printDefaultDeviceInfo();
        }
        
        benchmarkMatrixMultiplication();
        benchmarkLargeMatrixMultiplication(784, 128, 32);
    }
    
    private static void benchmarkMatrixMultiplication() {
        int m = 1500;
        int n = 1500;
        int p = 1500;
        
        System.out.println("\n=== Matrix multiplication (" + m + "x" + n + ") * (" + n + "x" + p + ") ===");

        TensorFactory.forceCPU();
        Tensor cpuA = TensorFactory.random(m, n);
        Tensor cpuB = TensorFactory.random(n, p);

        TensorFactory.useGPUIfAvailable();
        Tensor gpuA = TensorFactory.random(m, n);
        Tensor gpuB = TensorFactory.random(n, p);
        
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                gpuA.set(cpuA.get(i, j), i, j);
            }
        }
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                gpuB.set(cpuB.get(i, j), i, j);
            }
        }
        
        // CPU benchmark
        long cpuStart = System.nanoTime();
        Tensor cpuResult = cpuA.matmul(cpuB);
        long cpuEnd = System.nanoTime();
        
        // GPU benchmark
        long gpuStart = System.nanoTime();
        Tensor gpuResult = gpuA.matmul(gpuB);
        long gpuEnd = System.nanoTime();
        
        boolean resultsMatch = checkResultsMatch(cpuResult, gpuResult);

        double tookCPU = (cpuEnd - cpuStart) / 1e6;
        double tookGPU = (gpuEnd - gpuStart) / 1e6;
        double speedup = tookCPU / tookGPU;

        System.out.printf("CPU: %.4f ms\n", tookCPU);
        System.out.printf("GPU: %.4f ms\n", tookGPU);
        System.out.printf("Speedup: %.4fx\n", speedup);
        System.out.printf("Results match: %s\n", resultsMatch);
    }
    
    private static void benchmarkLargeMatrixMultiplication(int m, int n, int p) {
        System.out.println("\n=== Large matrix multiplication (" + m + "x" + n + ") * (" + n + "x" + p + ") ===");

        TensorFactory.forceCPU();
        System.out.println("Creating matrices...");
        Tensor cpuA = TensorFactory.ones(m, n);
        Tensor cpuB = TensorFactory.ones(n, p);

        TensorFactory.useGPUIfAvailable();
        Tensor gpuA = TensorFactory.ones(m, n);
        Tensor gpuB = TensorFactory.ones(n, p);
        
        System.out.println("Benchmark CPU in progress...");
        long cpuStart = System.nanoTime();
        Tensor cpuResult = cpuA.matmul(cpuB);
        long cpuEnd = System.nanoTime();
        
        System.out.println("Benchmark GPU in progress...");
        long gpuStart = System.nanoTime();
        Tensor gpuResult = gpuA.matmul(gpuB);
        long gpuEnd = System.nanoTime();

        boolean resultsMatch = checkResultsMatch(cpuResult, gpuResult);

        double tookCPU = (cpuEnd - cpuStart) / 1e6;
        double tookGPU = (gpuEnd - gpuStart) / 1e6;
        double speedup = tookCPU / tookGPU;

        System.out.printf("CPU: %.4f ms\n", tookCPU);
        System.out.printf("GPU: %.4f ms\n", tookGPU);
        System.out.printf("Speedup: %.4fx\n", speedup);
        System.out.printf("Results match: %s\n", resultsMatch);
    }
    
    private static boolean checkResultsMatch(Tensor a, Tensor b) {
        if (a.elements() != b.elements()) return false;
        if (!java.util.Arrays.equals(a.shape(), b.shape())) return false;
        
        double epsilon = 1e-5;  
        
        int[] shape = a.shape();
        
        if (shape.length == 1) {
            for (int i = 0; i < shape[0]; i++) {
                if (Math.abs(a.get(i) - b.get(i)) > epsilon) {
                    System.out.println("Mismatch at index " + i + ": " + a.get(i) + " vs " + b.get(i));
                    return false;
                }
            }
        } else if (shape.length == 2) {
            for (int i = 0; i < shape[0]; i++) {
                for (int j = 0; j < shape[1]; j++) {
                    if (Math.abs(a.get(i, j) - b.get(i, j)) > epsilon) {
                        System.out.println("Mismatch at [" + i + "," + j + "]: " + 
                                           a.get(i, j) + " vs " + b.get(i, j));
                        return false;
                    }
                }
            }
        } else {
            for (int i = 0; i < a.elements(); i++) {
                float valueA = getLinearElement(a, i);
                float valueB = getLinearElement(b, i);
                
                if (Math.abs(valueA - valueB) > epsilon) {
                    System.out.println("Mismatch at linear index " + i + ": " + valueA + " vs " + valueB);
                    return false;
                }
            }
        }
        
        return true;
    }
    
    private static float getLinearElement(Tensor tensor, int linearIndex) {
        int[] shape = tensor.shape();
        
        if (shape.length == 1) {
            return tensor.get(linearIndex);
        }
        else if (shape.length == 2) {
            int cols = shape[1];
            int row = linearIndex / cols;
            int col = linearIndex % cols;
            return tensor.get(row, col);
        } 
        else {
            int[] indices = new int[shape.length];
            int remaining = linearIndex;
            for (int i = shape.length - 1; i >= 0; i--) {
                indices[i] = remaining % shape[i];
                remaining /= shape[i];
            }
            return tensor.get(indices);
        }
    }
}
