package net.echo.examples.gpt;

import com.google.common.io.Files;
import net.echo.brain4j.Brain4J;
import net.echo.brain4j.layer.impl.transformers.EmbedLayer;
import net.echo.brain4j.layer.impl.transformers.PosEncodeLayer;
import net.echo.brain4j.layer.impl.transformers.TrDecoder;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.impl.Transformer;
import net.echo.brain4j.training.optimizer.impl.Adam;
import net.echo.brain4j.transformers.Vocabulary;
import net.echo.brain4j.transformers.tokenizer.Tokenizer;
import net.echo.brain4j.transformers.tokenizer.impl.SimpleTokenizer;
import net.echo.brain4j.layer.impl.transformers.VocabularyMapper;
import net.echo.math.tensor.Tensor;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

public class GPTExample {

    public static void main(String[] args) throws IOException {
        new GPTExample().start();
    }

    public void start() throws IOException {
        Brain4J.setLogging(true);

        int embeddingDim = 64;
        List<String> samples = Files.readLines(new File("dataset.txt"), StandardCharsets.UTF_8);

        Tokenizer tokenizer = new SimpleTokenizer();
        Vocabulary vocabulary = new Vocabulary(tokenizer, samples);

        Model model = new Transformer(
                new EmbedLayer(vocabulary.size(), embeddingDim),
                new PosEncodeLayer(embeddingDim, 1024),

                new TrDecoder(4, embeddingDim),
                new VocabularyMapper(vocabulary.size(), embeddingDim, 1.0)
        );

        System.out.println("Loaded vocabulary of " + vocabulary.size() + " tokens.");
        model.compile(Loss.CROSS_ENTROPY, new Adam(0.1));

        String phrase = "hello!";

        for (int i = 0; i < 10; i++) {
            Tensor input = tokenizer.tokenize(vocabulary, phrase);

            Tensor nextToken = model.predict(input);
            String word = vocabulary.getToken(nextToken.argmax());

            phrase += word;
            System.out.print(word);
        }
    }
}
