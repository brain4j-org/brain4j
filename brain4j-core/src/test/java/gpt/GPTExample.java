package gpt;

import org.brain4j.core.Brain4J;
import org.brain4j.core.layer.impl.transformers.EmbedLayer;
import org.brain4j.core.layer.impl.transformers.PosEncodeLayer;
import org.brain4j.core.layer.impl.transformers.TrDecoder;
import org.brain4j.core.layer.impl.transformers.VocabularyMapper;
import org.brain4j.core.loss.Loss;
import org.brain4j.core.model.Model;
import org.brain4j.core.model.impl.Transformer;
import org.brain4j.core.training.optimizer.impl.Lion;
import org.brain4j.core.transformers.Vocabulary;
import org.brain4j.core.transformers.tokenizer.Tokenizer;
import org.brain4j.core.transformers.tokenizer.impl.SimpleTokenizer;
import org.brain4j.math.data.ListDataSource;
import org.brain4j.math.data.Sample;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

public class GPTExample {

    public static void main(String[] args) throws Exception {
        new GPTExample().start();
    }

    public void start() throws Exception {
        int embeddingDim = 64;

        Brain4J.setLogging(true);
        List<String> corpus = Files.readAllLines(new File("corpus.txt").toPath());

        Tokenizer tokenizer = new SimpleTokenizer();
        Vocabulary vocabulary = new Vocabulary(tokenizer, corpus);

        int vocabSize = vocabulary.size();

        Model model = new Transformer(
                new EmbedLayer(vocabSize, embeddingDim),
                new PosEncodeLayer(embeddingDim),

                new TrDecoder(2, embeddingDim),
                new VocabularyMapper(vocabSize, embeddingDim, 0.5)
        );

        model.compile(Loss.CROSS_ENTROPY, new Lion(0.001, 0.9));

        System.out.println(model.summary());
        System.out.println("Vocabulary Size: " + vocabSize);

        List<Sample> samples = new ArrayList<>();
        List<String> tokens = tokenizer.split(corpus.getFirst());

        StringBuilder text = new StringBuilder();
        int limit = tokens.size() - 1;

        for (int i = 0; i < limit; i++) {
            text.append(tokens.get(i));
            String output = tokens.get(i + 1);

            Tensor inputTensor = tokenizer.tokenize(vocabulary, text.toString());
            Tensor outputTensor = Tensors.create(vocabSize);

            int id = vocabulary.getId(output);
            outputTensor.set(1, id);

            samples.add(new Sample(inputTensor, outputTensor));
        }

        ListDataSource source = new ListDataSource(samples, false, 1);
        model.fit(source, 100, 1);

        String input = "The cat (Felis catus), also referred to as the domestic cat or";

        Tensor tensor = tokenizer.tokenize(vocabulary, input);
        Tensor output = model.predict(tensor).vector();

        String nextToken = vocabulary.getToken(output.argmax());
        System.out.println(input + nextToken);

        model.save("gpt.b4j");
    }
}
