package backend;

import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.impl.TensorNative;

public class TensorBackend {

    public static void main(String[] args) {
        int m = 1000;
        int n = 2000;
        int p = 3000;
        Tensor a = new TensorNative(m, n).fill(Math::random);
        Tensor b = new TensorNative(n, p).fill(Math::random);

        Tensor cpuA = a.cpu();
        Tensor cpuB = b.cpu();

        Tensor gpuA = a.gpu();
        Tensor gpuB = b.gpu();

        long nativeStart = System.nanoTime();
        a.matmul(b);
        long nativeEnd = System.nanoTime();

        long cpuStart = System.nanoTime();
        cpuA.matmul(cpuB);
        long cpuEnd = System.nanoTime();

        long gpuStart = System.nanoTime();
        gpuA.matmul(gpuB);
        long gpuEnd = System.nanoTime();

        double tookNative = (nativeEnd - nativeStart) / 1e6;
        double tookCPU = (cpuEnd - cpuStart) / 1e6;
        double tookGPU = (gpuEnd - gpuStart) / 1e6;
        System.out.println("Native took " + tookNative + " ms");
        System.out.println("CPU took " + tookCPU + " ms");
        System.out.println("GPU took " + tookGPU + " ms");
    }
}
