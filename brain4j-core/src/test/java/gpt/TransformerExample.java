package gpt;

import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.impl.Transformer;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.transformers.TransformerDecoder;
import net.echo.brain4j.transformers.TransformerEncoder;
import net.echo.brain4j.transformers.Vocabulary;
import net.echo.brain4j.transformers.VocabularyMapper;
import net.echo.brain4j.transformers.encoding.PositionalEncoding;
import net.echo.math4j.BrainUtils;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.vector.Vector;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public class TransformerExample {

    private final int dimension = 16;
    private final PositionalEncoding encoding = new PositionalEncoding(100, dimension);

    public static void main(String[] args) throws IOException {
        TransformerExample example = new TransformerExample();
        example.start();
    }

    public List<String> getExamples() throws IOException {
        return Files.readAllLines(Path.of("dataset.txt"));
    }

    public void start() throws IOException {
        List<String> examples = getExamples();

        Vocabulary vocabulary = new Vocabulary(examples, dimension);
        vocabulary.tokenize();

        int vocabSize = vocabulary.getVocabSize();
        int dimension = 16;

        System.out.println("Vocabulary size: " + vocabSize);
        // TensorFactory.useGPUIfAvailable();

        Transformer transformer = new Transformer(
                new TransformerEncoder(4, dimension),
                new TransformerDecoder(4, dimension),
                new VocabularyMapper(vocabSize, dimension, 11)
        );

        transformer.compile(LossFunctions.CROSS_ENTROPY, new Adam(0.001));

        System.out.println(transformer.getStats());

        String phrase = "hello, how are you?";
        StringBuilder response = new StringBuilder();

        for (int i = 0; i < 5; i++) {
            System.out.println("User: " + phrase);

            Tensor input = vocabulary.encode(phrase);
            Tensor encoded = encoding.encode(input);

            Tensor output = transformer.predict(encoded);

            System.out.println("===== Probability Distribution =====");
            System.out.println(output.toString("%.3f"));

            int indexOfMax = BrainUtils.indexOfMaxValue(Vector.of(output.toArray()));

            String word = vocabulary.indexToWord(indexOfMax);
            System.out.println("Chat Bot: " + response);

            response.append(word).append(" ");
            phrase += " " + word;
        }
    }
}
