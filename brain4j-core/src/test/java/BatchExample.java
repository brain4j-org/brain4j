import net.echo.brain4j.Brain4J;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.Tensors;

public class BatchExample {

    public static void main(String[] args) {
        Brain4J.useGPUIfAvailable();

        int batches = 8;
        int samples = 327680;
        int batchSize = samples / batches;

        System.out.println("Batch size: " + batchSize);
        System.out.println("Batches: " + batches);

        Tensor a = Tensors.random(batchSize, 13);

        Tensor b = Tensors.random(13, 256);
        Tensor c = Tensors.random(256, 256);
        Tensor d = Tensors.random(256, 256);
        Tensor e = Tensors.random(256, 1);

        long start = System.nanoTime();

        for (int i = 0; i < batches; i++) {
            a.matmul(b).matmul(c).matmul(d).matmul(e);
        }

        long end = System.nanoTime();

        System.out.println("Time: " + (end - start) / 1e6 + " ms");
    }
}
