package gpt;

import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.model.impl.Transformer;
import net.echo.brain4j.structure.StatesCache;
import net.echo.brain4j.training.optimizer.impl.GradientDescent;
import net.echo.brain4j.transformers.DecoderGroup;
import net.echo.brain4j.transformers.vocabulary.VocabularyMapper;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

public class BigTransformer {

    public static void main(String[] args) {
        new BigTransformer().start();
    }

    void start() {
        int dimension = 1536;
        int numHeads = 16;
        int vocabSize = 50000;

        var model = new Transformer(
                new DecoderGroup(16, numHeads, dimension),

                new VocabularyMapper(vocabSize, dimension, 5)
        );

        System.out.println("Compiling...");

        long start = System.nanoTime();
        model.compile(Loss.CROSS_ENTROPY, new GradientDescent(0.01));
        double took = (System.nanoTime() - start) / 1e6;

        System.out.println(model.summary());
        System.out.printf("Took %s ms to initialize%n", took);

        Tensor input = TensorFactory.random(3, dimension);

        System.out.println("\n--- First prediction (without KV cache) ---");
        start = System.nanoTime();
        
        model.predict(input);
        
        took = (System.nanoTime() - start) / 1e6;
        System.out.printf("Time for first prediction: %.2f ms%n", took);

        System.out.println("\n--- Sequential predictions (with KV cache) ---");
        
        StatesCache sharedCache = new StatesCache();
        sharedCache.markAsNewSession();
        
        start = System.nanoTime();
        
        model.predict(sharedCache, input, false);
        
        took = (System.nanoTime() - start) / 1e6;
        System.out.printf("Time for first prediction (with cache): %.2f ms%n", took);
        
        Tensor newInput = TensorFactory.zeros(4, dimension);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < dimension; j++) {
                newInput.set(input.get(i, j), i, j);
            }
        }
        Tensor newToken = TensorFactory.random(1, dimension);
        for (int j = 0; j < dimension; j++) {
            newInput.set(newToken.get(0, j), 3, j);
        }
        
        start = System.nanoTime();
        
        model.predict(sharedCache, newInput, false);
        
        took = (System.nanoTime() - start) / 1e6;
        System.out.printf("Time for second prediction (with cache): %.2f ms%n", took);
        
        Tensor finalInput = TensorFactory.zeros(5, dimension);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < dimension; j++) {
                finalInput.set(newInput.get(i, j), i, j);
            }
        }

        Tensor anotherToken = TensorFactory.random(1, dimension);
        for (int j = 0; j < dimension; j++) {
            finalInput.set(anotherToken.get(0, j), 4, j);
        }
        
        start = System.nanoTime();
        
        model.predict(sharedCache, finalInput, false);
        
        took = (System.nanoTime() - start) / 1e6;
        System.out.printf("Time for third prediction (with cache): %.2f ms%n", took);
        
        System.out.println("\n--- New prompt (reset of KV cache) ---");
        Tensor newPrompt = TensorFactory.random(2, dimension);
        
        sharedCache.markAsNewSession();
        
        start = System.nanoTime();
        
        model.predict(sharedCache, newPrompt, false);
        
        took = (System.nanoTime() - start) / 1e6;
        System.out.printf("Time for new session: %.2f ms%n", took);

        System.out.println("Took " + took + " ms to predict");
    }
}
