package gpt;

import net.echo.brain4j.Brain4J;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.model.impl.Transformer;
import net.echo.brain4j.structure.StatesCache;
import net.echo.brain4j.training.optimizer.impl.GradientDescent;
import net.echo.brain4j.transformers.ContextWindow;
import net.echo.brain4j.transformers.group.DecoderGroup;
import net.echo.brain4j.transformers.vocabulary.VocabularyMapper;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

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
                new DecoderGroup(16, numHeads, dimension),
                new VocabularyMapper(vocabSize, dimension, 5)
        );

        System.out.println("Compiling...");

        double took = evaluate(() -> model.compile(Loss.CROSS_ENTROPY, new GradientDescent(0.01)));

        System.out.println(model.summary());
        System.out.printf("Took %s ms to initialize%n", took);

        ContextWindow window = new ContextWindow(100, dimension);
        window.append(TensorFactory.zeros(dimension));

        StatesCache sharedCache = new StatesCache();

        for (int i = 0; i < 10; i++) {
            Tensor finalInput = window.toInput();

            long start = System.nanoTime();
            Tensor output = model.predict(sharedCache, finalInput, false);

            took = (System.nanoTime() - start) / 1e6;
            System.out.printf("Took %s ms to predict%n", took);

            window.append(output);
        }
    }

    private double evaluate(Runnable runnable) {
        long start = System.nanoTime();
        runnable.run();
        return (System.nanoTime() - start) / 1e6;
    }
}
