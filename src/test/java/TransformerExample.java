import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.impl.Transformer;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.transformers.TransformerEncoder;
import net.echo.brain4j.utils.Vector;

import java.util.ArrayList;
import java.util.List;

public class TransformerExample {

    public static void main(String[] args) throws InterruptedException {
        TransformerExample example = new TransformerExample();
        example.start();
    }

    public void start() throws InterruptedException {
        Transformer transformer = new Transformer(
                new TransformerEncoder(4, 784, 1.0),
                new TransformerEncoder(4, 784, 1.0),
                new TransformerEncoder(4, 784, 1.0),
                new TransformerEncoder(4, 784, 1.0)
        );

        transformer.compile(LossFunctions.BINARY_CROSS_ENTROPY, new Adam(0.001));

        System.out.println(transformer.getStats());

        List<Vector> vectors = new ArrayList<>();

        for (int i = 0; i < 10; i++) {
            vectors.add(Vector.random(784));
        }

        long start = System.nanoTime();
        var output = transformer.predict(vectors);
        double took = (System.nanoTime() - start) / 1e6;

        System.out.println("Took: " + took + " ms");

        Thread.sleep(Integer.MAX_VALUE);
    }
}
