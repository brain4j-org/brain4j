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
import net.echo.math.data.ListDataSource;
import net.echo.math.data.Sample;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.Tensors;

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

public class GPTExample {

    public static void main(String[] args) throws Exception {
        new GPTExample().start();
    }

    public void start() throws Exception {
        int embeddingDim = 32;

        Brain4J.setLogging(true);
        List<String> corpus = Files.readAllLines(new File("corpus.txt").toPath());

        Tokenizer tokenizer = new SimpleTokenizer();
        Vocabulary vocabulary = new Vocabulary(tokenizer, corpus);

        int vocabSize = vocabulary.size();

        Model model = new Transformer(
                new EmbedLayer(vocabSize, embeddingDim),
                new PosEncodeLayer(embeddingDim),

                new TrDecoder(8, embeddingDim),
                new TrDecoder(8, embeddingDim),

                new VocabularyMapper(vocabSize, embeddingDim, 0.1)
        );

        model.compile(Loss.CROSS_ENTROPY, new Lion(0.01, 0.9));


        System.out.println(model.summary());
        System.out.println("Vocabulary Size: " + vocabSize);

        List<Sample> samples = new ArrayList<>();
        List<String> tokens = tokenizer.split(corpus.getFirst());

        StringBuilder text = new StringBuilder();
        int limit = 18; // tokens.size() - 1;

        for (int i = 0; i < limit; i++) {
            text.append(tokens.get(i));
            String output = tokens.get(i + 1);

            Tensor inputTensor = tokenizer.tokenize(vocabulary, text.toString());
            Tensor outputTensor = Tensors.create(vocabSize);

            int id = vocabulary.getId(output);
            outputTensor.set(1, id);

            samples.add(new Sample(inputTensor, outputTensor));
        }

        System.out.println("Last Sample: " + text);

        ListDataSource source = new ListDataSource(samples, false, 1);
        model.fit(source, 5000, 100);

        String input = "The cat (Felis catus), also referred to as the domestic cat or";

        Tensor tensor = tokenizer.tokenize(vocabulary, input);
        Tensor output = model.predict(tensor).vector();

        System.out.println("Input tensor -> " + tensor.toString("%.1f"));
        System.out.println("Output tensor -> " + output.toString("%.4f"));

        String nextToken = vocabulary.getToken(output.argmax());
        System.out.println(input + nextToken);

        model.save("gpt.b4j");
    }
}
