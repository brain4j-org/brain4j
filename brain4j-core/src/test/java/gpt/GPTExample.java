package gpt;

import net.echo.brain4j.Brain4J;
import net.echo.brain4j.layer.impl.transformers.EmbedLayer;
import net.echo.brain4j.layer.impl.transformers.PosEncodeLayer;
import net.echo.brain4j.layer.impl.transformers.TrDecoder;
import net.echo.brain4j.layer.impl.transformers.VocabularyMapper;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.impl.Transformer;
import net.echo.brain4j.training.optimizer.impl.Lion;
import net.echo.brain4j.transformers.Vocabulary;
import net.echo.brain4j.transformers.tokenizer.Tokenizer;
import net.echo.brain4j.transformers.tokenizer.impl.SimpleTokenizer;
import net.echo.math.tensor.Tensor;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;

public class GPTExample {

    public static void main(String[] args) throws IOException {
        new GPTExample().start();
    }

    public void start() throws IOException {
        Brain4J.setLogging(true);
        final int embeddingDim = 64;

        List<String> corpus = Files.readAllLines(new File("corpus.txt").toPath());

        Tokenizer tokenizer = new SimpleTokenizer();
        Vocabulary vocabulary = new Vocabulary(tokenizer, corpus);

        Model model = new Transformer(
                new EmbedLayer(vocabulary.size(), embeddingDim),
                new PosEncodeLayer(embeddingDim),

                new TrDecoder(8, embeddingDim),
                new TrDecoder(8, embeddingDim),

                new VocabularyMapper(vocabulary.size(), embeddingDim, 3)
        );

        model.compile(Loss.CROSS_ENTROPY, new Lion(0.01, 0.9));
        System.out.println(model.summary());

        String input = "hello, world!";

        Tensor tensor = tokenizer.tokenize(vocabulary, input);
        Tensor output = model.predict(tensor).vector();

        System.out.println("Input tensor -> " + tensor.toString("%.1f"));
        System.out.println("Output tensor -> " + output.toString("%.4f"));
    }
}
