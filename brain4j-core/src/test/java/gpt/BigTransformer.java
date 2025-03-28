package gpt;

import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.model.impl.Transformer;
import net.echo.brain4j.training.optimizer.impl.Adam;
import net.echo.brain4j.training.optimizer.impl.GradientDescent;
import net.echo.brain4j.transformers.DecoderGroup;
import net.echo.brain4j.transformers.TransformerDecoder;
import net.echo.brain4j.transformers.vocabulary.VocabularyMapper;

public class BigTransformer {

    public static void main(String[] args) {
        new BigTransformer().start();
    }

    void start() {
        int dimension = 1536;
        int numHeads = 16;
        int vocabSize = 50000;

        var model = new Transformer(
                new DecoderGroup(32, numHeads, dimension),

                new VocabularyMapper(vocabSize, dimension, 5)
        );

        System.out.println("Compiling...");

        long start = System.nanoTime();
        model.compile(Loss.CROSS_ENTROPY, new GradientDescent(0.01));
        double took = (System.nanoTime() - start) / 1e6;

        System.out.println(model.summary());
        System.out.printf("Took %s ms to initialize", took);
    }
}
