package convolve;

import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;
import net.echo.math4j.math.tensor.impl.TensorGPU;
import net.echo.math4j.math.tensor.ops.Convolution;

import java.util.Arrays;

public class ConvolveTest {
    
    private static int passedTests = 0;
    private static int failedTests = 0;
    
    public static void main(String[] args) {
        System.out.println("Execution of convolution tests...\n");
        
        testConvolve1DSimple();
        
        testConvolve2DSimple();
        testGaussianBlur2D();
        
        testCompareDirectAndFFT();
        
        if (TensorGPU.isGpuAvailable()) {
            testConvolutionGPU1D();
            testConvolutionGPU2D();
            testPerformanceComparison();
        } else {
            System.out.println("Test GPU skipped: GPU not available");
        }
        
        System.out.println("\nTest results:");
        System.out.println("Passed tests: " + passedTests);
        System.out.println("Failed tests: " + failedTests);
        
        if (failedTests > 0) {
            System.exit(1);
        }
    }
    
    private static void testConvolve1DSimple() {
        System.out.println("Test: simple 1D convolution");
        try {
            Tensor input = TensorFactory.of(new int[]{5}, 1, 2, 3, 4, 5);
            
            Tensor kernel = TensorFactory.of(new int[]{3}, 1, 2, 1);
            
            Tensor result = input.convolve(kernel);
            
            assertArrayEquals("Shape", new int[]{5}, result.shape());
            
            assertEquals("Value[0]", 4.0f, result.get(0), 1e-5f);
            assertEquals("Value[1]", 8.0f, result.get(1), 1e-5f);
            assertEquals("Value[2]", 12.0f, result.get(2), 1e-5f);
            assertEquals("Value[3]", 16.0f, result.get(3), 1e-5f);
            assertEquals("Value[4]", 14.0f, result.get(4), 1e-5f);
            
            passTest();
        } catch (Exception e) {
            failTest("testConvolve1DSimple", e);
        }
    }

    private static void testConvolve2DSimple() {
        System.out.println("Test: simple 2D convolution");
        try {
            Tensor input = TensorFactory.matrix(3, 3,
                    1, 2, 3,
                    4, 5, 6,
                    7, 8, 9);
            
            Tensor kernel = TensorFactory.matrix(3, 3,
                    0, 0, 0,
                    0, 1, 0,
                    0, 0, 0);
            
            Tensor result = input.convolve(kernel);
            
            assertArrayEquals("Shape", new int[]{3, 3}, result.shape());
            
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    assertEquals("Value[" + i + "," + j + "]", input.get(i, j), result.get(i, j), 1e-5f);
                }
            }
            
            passTest();
        } catch (Exception e) {
            failTest("testConvolve2DSimple", e);
        }
    }

    private static void testGaussianBlur2D() {
        System.out.println("Test: Gaussian blur 2D");
        try {
            float[][] inputArray = new float[5][5];
            inputArray[2][2] = 1.0f;
            
            Tensor input = TensorFactory.of(new int[]{5, 5}, flattenArray(inputArray));
            
            float[][] kernelArray = new float[][]{
                {1/16f, 2/16f, 1/16f},
                {2/16f, 4/16f, 2/16f},
                {1/16f, 2/16f, 1/16f}
            };
            
            Tensor kernel = TensorFactory.of(new int[]{3, 3}, flattenArray(kernelArray));
            
            Tensor result = input.convolve(kernel);
            
            assertArrayEquals("Shape", new int[]{5, 5}, result.shape());
            
            assertEquals("Central value", 0.25f, result.get(2, 2), 1e-5f);
            
            assertEquals("Adjacent value", 0.125f, result.get(1, 2), 1e-5f);
            assertEquals("Adjacent value", 0.125f, result.get(3, 2), 1e-5f);
            assertEquals("Adjacent value", 0.125f, result.get(2, 1), 1e-5f);
            assertEquals("Adjacent value", 0.125f, result.get(2, 3), 1e-5f);
            
            assertEquals("Diagonal value", 0.0625f, result.get(1, 1), 1e-5f);
            assertEquals("Diagonal value", 0.0625f, result.get(1, 3), 1e-5f);
            assertEquals("Diagonal value", 0.0625f, result.get(3, 1), 1e-5f);
            assertEquals("Diagonal value", 0.0625f, result.get(3, 3), 1e-5f);
            
            passTest();
        } catch (Exception e) {
            failTest("testGaussianBlur2D", e);
        }
    }
    
    private static void testCompareDirectAndFFT() {
        System.out.println("Test: comparison between direct and FFT convolution");
        try {
            int size = 32;
            float[] inputData = new float[size];
            for (int i = 0; i < size; i++) {
                inputData[i] = i % 5;
            }
            Tensor input = TensorFactory.of(new int[]{size}, inputData);
            
            Tensor kernel = TensorFactory.of(new int[]{3}, new float[]{1.0f, 1.0f, 1.0f});
            
            Tensor directResult = Convolution.convolve1D(
                input, kernel, Convolution.PaddingMode.SAME, Convolution.ConvolutionType.DIRECT);
            
            Tensor fftResult = Convolution.convolve1D(
                input, kernel, Convolution.PaddingMode.SAME, Convolution.ConvolutionType.FFT);
            
            boolean allClose = true;
            for (int i = 0; i < size && allClose; i++) {
                float diff = Math.abs(directResult.get(i) - fftResult.get(i));
                if (diff > 1e-1f) {
                    allClose = false;
                    System.out.println("  Excessive difference at index " + i + 
                                     ": direct=" + directResult.get(i) + 
                                     ", FFT=" + fftResult.get(i));
                }
            }
            
            assertTrue("The results of direct and FFT convolution are similar", allClose);
            
            passTest();
        } catch (Exception e) {
            failTest("testCompareDirectAndFFT", e);
        }
    }
    
    private static void testConvolutionGPU1D() {
        System.out.println("Test: convolution 1D on GPU");
        try {
            Tensor input = TensorGPU.of(new int[]{5}, new float[]{1, 2, 3, 4, 5});
            
            Tensor kernel = TensorGPU.of(new int[]{3}, new float[]{1, 2, 1});
            
            Tensor result = input.convolve(kernel);
            
            assertArrayEquals("Shape", new int[]{5}, result.shape());
            
            assertEquals("Value[0]", 4.0f, result.get(0), 1e-3f);
            assertEquals("Value[1]", 8.0f, result.get(1), 1e-3f);
            assertEquals("Value[2]", 12.0f, result.get(2), 1e-3f);
            assertEquals("Value[3]", 16.0f, result.get(3), 1e-3f);
            assertEquals("Value[4]", 14.0f, result.get(4), 1e-3f);
            
            passTest();
        } catch (Exception e) {
            failTest("testConvolutionGPU1D", e);
        }
    }

    private static void testConvolutionGPU2D() {
        System.out.println("Test: convolution 2D on GPU");
        try {
            Tensor input = TensorGPU.of(new int[]{3, 3}, new float[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
            });
            
            Tensor kernel = TensorGPU.of(new int[]{3, 3}, new float[]{
                0, 0, 0,
                0, 1, 0,
                0, 0, 0
            });
            
            Tensor result = input.convolve(kernel);
            
            assertArrayEquals("Shape", new int[]{3, 3}, result.shape());
            
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    assertEquals("Value[" + i + "," + j + "]", input.get(i, j), result.get(i, j), 1e-3f);
                }
            }
            
            passTest();
        } catch (Exception e) {
            failTest("testConvolutionGPU2D", e);
        }
    }

    private static void testPerformanceComparison() {
        System.out.println("\nTest: CPU vs GPU performance comparison");
        try {
            if (!TensorGPU.isGpuAvailable()) {
                System.out.println("  GPU not available, test skipped");
                passTest();
                return;
            }
            
            int size = 256;
            System.out.println("  Tensor size: " + size + "x" + size);
            
            Tensor inputCPU = TensorFactory.random(size, size);
            Tensor inputGPU = inputCPU.gpu();

            Tensor kernelCPU = TensorFactory.matrix(3, 3,
                    -1, -1, -1,
                    -1, 8, -1,
                    -1, -1, -1);
            Tensor kernelGPU = kernelCPU.gpu();
            
            System.out.println("  Executing convolution on CPU...");
            long startCPU = System.nanoTime();
            Tensor resultCPU = inputCPU.convolve(kernelCPU);
            long cpuTime = System.nanoTime() - startCPU;
            
            System.out.println("  Executing convolution on GPU...");
            long startGPU = System.nanoTime();
            Tensor resultGPU = inputGPU.convolve(kernelGPU);
            long gpuTime = System.nanoTime() - startGPU;
            
            System.out.println("  CPU time: " + (cpuTime / 1e6) + " ms");
            System.out.println("  GPU time: " + (gpuTime / 1e6) + " ms");
            double speedup = (double)cpuTime / gpuTime;
            System.out.println("  Speedup: " + String.format("%.2f", speedup) + "x");
            
            assertArrayEquals("Shape", resultCPU.shape(), resultGPU.shape());
            
            float maxDiff = 0;
            int errorRow = -1, errorCol = -1;

            // the maximum difference allowed between the CPU and GPU results.
            // it determines whether if the algorithm is correct.
            double allowedDiff = 0.5f; 

            for (int i = 0; i < resultCPU.shape()[0]; i++) {
                for (int j = 0; j < resultCPU.shape()[1]; j++) {
                    float diff = Math.abs(resultCPU.get(i, j) - resultGPU.get(i, j));
                    if (diff > maxDiff) {
                        maxDiff = diff;
                        errorRow = i;
                        errorCol = j;
                    }
                }
            }

            System.out.println("  Maximum difference: " + maxDiff + " at position [" + errorRow + "," + errorCol + "]");

            assertTrue("Results within acceptable tolerance", maxDiff < allowedDiff);

            passTest();
        } catch (Exception e) {
            failTest("testPerformanceComparison", e);
        }
    }
 
    private static float[] flattenArray(float[][] array) {
        int rows = array.length;
        int cols = (rows > 0) ? array[0].length : 0;
        float[] flattened = new float[rows * cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flattened[i * cols + j] = array[i][j];
            }
        }
        
        return flattened;
    }

    private static void assertEquals(String message, float expected, float actual, float delta) {
        if (Math.abs(expected - actual) > delta) {
            throw new AssertionError(message + ": expected <" + expected + ">, got <" + actual + ">");
        }
    }
    
    private static void assertArrayEquals(String message, int[] expected, int[] actual) {
        if (!Arrays.equals(expected, actual)) {
            throw new AssertionError(
                message + ": expected <" + Arrays.toString(expected) + ">, got <" + 
                Arrays.toString(actual) + ">");
        }
    }
    
    private static void assertTrue(String message, boolean condition) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }
    
    private static void passTest() {
        System.out.println("  ✓ Success");
        passedTests++;
    }
    
    private static void failTest(String testName, Exception e) {
        System.out.println("  ✗ Failure in " + testName + ": " + e.getMessage());
        e.printStackTrace(System.out);
        failedTests++;
    }
}
