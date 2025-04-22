package largegpt;

import net.echo.brain4j.Brain4J;
import net.echo.brain4j.layer.impl.transformers.EmbedLayer;
import net.echo.brain4j.layer.impl.transformers.PosEncodeLayer;
import net.echo.brain4j.layer.impl.transformers.VocabularyMapper;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.model.impl.Transformer;
import net.echo.brain4j.structure.StatesCache;
import net.echo.brain4j.training.optimizer.impl.Adam;
import net.echo.brain4j.transformers.group.DecoderGroup;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.TensorFactory;

public class BigTransformerExample {

    public static void main(String[] args) {
        new BigTransformerExample().start();
    }

    private void start() {
        Brain4J.setLogging(true);
        Brain4J.useGPUIfAvailable();

        int dimension = 1536;
        int numHeads = 16;
        int vocabSize = 50000;

        Transformer model = new Transformer(
                new EmbedLayer(vocabSize, dimension),
                new PosEncodeLayer(dimension),

                new DecoderGroup(16, numHeads, dimension),
                new VocabularyMapper(vocabSize, dimension, 5)
        );

        System.out.println("Compiling...");

        double took = evaluate(() -> model.compile(Loss.CROSS_ENTROPY, new Adam(0.01)));

        System.out.println(model.summary());
        System.out.printf("Took %s ms to initialize%n", took);

        Tensor inputs = TensorFactory.vector(10)
                .mapWithIndex((i, x) -> Float.valueOf(i));

        StatesCache sharedCache = new StatesCache();
        double totalTime = 0.0;

        for (int i = 0; i < 10; i++) {
            Tensor finalInputs = inputs;

            took = evaluate(() -> model.predict(sharedCache, finalInputs));
            totalTime += took;

            double mean = totalTime / (i + 1);
            System.out.printf("Took %.4f ms to predict - %.4f ms on average %n", took, mean);

            inputs = TensorFactory.vector(inputs.elements() + 1)
                    .mapWithIndex((j, x) -> Float.valueOf(j));
        }
    }

    private double evaluate(Runnable runnable) {
        long start = System.nanoTime();
        runnable.run();
        return (System.nanoTime() - start) / 1e6;
    }
}
