import net.echo.brain4j.Brain4J;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.Tensors;

public class BatchExample {

    public static void main(String[] args) {
        int batchSize = 8192 * 5;
        int samples = 307200;

        int batches = samples / batchSize;

        System.out.println("Batch size: " + batchSize);
        System.out.println("Batches: " + batches);
        Brain4J.useGPUIfAvailable();

        Tensor a = Tensors.matrix(batchSize, 13);

        Tensor b = Tensors.matrix(13, 256);
        Tensor c = Tensors.matrix(256, 256);
        Tensor d = Tensors.matrix(256, 256);
        Tensor e = Tensors.matrix(256, 1);

        long start = System.nanoTime();

        for (int i = 0; i < batches; i++) {
            a.matmul(b).matmul(c).matmul(d).matmul(e);
        }

        long end = System.nanoTime();

        System.out.println("Time: " + (end - start) / 1e6 + " ms");
    }
}
