package gpt;

import net.echo.brain4j.Brain4J;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.model.impl.Transformer;
import net.echo.brain4j.structure.StatesCache;
import net.echo.brain4j.training.optimizer.impl.GradientDescent;
import net.echo.brain4j.transformers.DecoderGroup;
import net.echo.brain4j.transformers.vocabulary.VocabularyMapper;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

import java.util.List;

public class BigTransformer {

    public static void main(String[] args) {
        new BigTransformer().start();
    }

    private void start() {
        Brain4J.setLogging(true);
        int dimension = 1536;
        int numHeads = 16;
        int vocabSize = 50000;

        var model = new Transformer(
                new DecoderGroup(16, numHeads, dimension),
                new VocabularyMapper(vocabSize, dimension, 5)
        );

        System.out.println("Compiling...");

        double took = evaluate(() -> model.compile(Loss.CROSS_ENTROPY, new GradientDescent(0.01)));

        System.out.println(model.summary());
        System.out.printf("Took %s ms to initialize%n", took);

        Tensor input = TensorFactory.random(1, dimension);

        StatesCache sharedCache = new StatesCache();

        for (int i = 0; i < 10; i++) {
            Tensor finalInput = input;
            took = evaluate(() -> model.predict(sharedCache, finalInput, false));
            System.out.printf("Took %s ms to predict%n", took);

            List<Tensor> tokens = TensorFactory.toList(input);
            tokens.add(TensorFactory.random(1, dimension));

            input = TensorFactory.mergeTensors(tokens);
        }
    }

    private double evaluate(Runnable runnable) {
        long start = System.nanoTime();
        runnable.run();
        return (System.nanoTime() - start) / 1e6;
    }
}
