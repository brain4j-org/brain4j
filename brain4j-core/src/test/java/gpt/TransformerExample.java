package gpt;

import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.impl.Transformer;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.transformers.TransformerDecoder;
import net.echo.brain4j.transformers.TransformerEncoder;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class TransformerExample {

    public static void main(String[] args) {
        TransformerExample example = new TransformerExample();
        example.start();
    }

    public List<String> getExamples() throws IOException {
        return Files.readAllLines(Path.of("dataset.txt"));
    }

    public void start() {
        int dimension = 784;
        TensorFactory.useGPUIfAvailable();

        Transformer transformer = new Transformer(
                new TransformerEncoder(4, dimension, 1.0),
                new TransformerEncoder(4, dimension, 1.0),
                new TransformerEncoder(4, dimension, 1.0),
                new TransformerEncoder(4, dimension, 1.0)
        );

        transformer.compile(LossFunctions.BINARY_CROSS_ENTROPY, new Adam(0.001));

        System.out.println(transformer.getStats());

        List<Tensor> inputs = new ArrayList<>();

        for (int i = 0; i < 10; i++) {
            inputs.add(TensorFactory.random(1, dimension));
        }

        long start = System.nanoTime();
        List<Tensor> output = transformer.predict(inputs);
        double took = (System.nanoTime() - start) / 1e6;

        System.out.println("Took " + took + " ms");
    }
}
