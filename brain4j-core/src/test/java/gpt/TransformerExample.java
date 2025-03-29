package gpt;

import net.echo.brain4j.Brain4J;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.model.impl.Transformer;
import net.echo.brain4j.structure.StatesCache;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.optimizer.impl.Adam;
import net.echo.brain4j.transformers.TransformerDecoder;
import net.echo.brain4j.transformers.vocabulary.Vocabulary;
import net.echo.brain4j.transformers.vocabulary.VocabularyMapper;
import net.echo.brain4j.transformers.encoding.PositionalEncoding;
import net.echo.math4j.BrainUtils;
import net.echo.math4j.DataSet;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class TransformerExample {

    private static final int EMBEDDING_SIZE = 64;
    private static final PositionalEncoding ENCODING = new PositionalEncoding(100, EMBEDDING_SIZE);

    public static void main(String[] args) throws Exception {
        new TransformerExample().start();
    }

    private List<String> loadExamples() throws Exception {
        return Files.readAllLines(Path.of("dataset.txt"));
    }

    private void start() throws Exception {
        Brain4J.setLogging(true);
        List<String> examples = loadExamples();

        Vocabulary vocabulary = new Vocabulary(examples, EMBEDDING_SIZE);
        vocabulary.tokenize();

        Transformer model = new Transformer(
                new TransformerDecoder(1, EMBEDDING_SIZE),
                new VocabularyMapper(vocabulary.getVocabSize(), EMBEDDING_SIZE, 5)
        );

        model.compile(Loss.CROSS_ENTROPY, new Adam(0.1));

        System.out.println(model.summary());
        System.out.println("Vocabulary size: " + vocabulary.getVocabSize());

        Scanner scanner = new Scanner(System.in);

        Map<String, String> samples = Map.of(
                "write a story", "Once upon a time, there was a small cat named Mia. Mia lived in a cozy house with Sarah. Every day, Mia played in the garden and chased butterflies. One day, she found a shiny key. It opened a hidden room full of toys. Mia was very happy!"
        );

        trainModel(vocabulary, samples, model);

        String prompt;

        do {
            System.out.print("Enter a prompt: ");
            prompt = scanner.nextLine() + " ";

            generateResponse(vocabulary, model, prompt);
            System.out.println();
        } while (!prompt.equals("end"));
    }

    private void trainModel(Vocabulary vocabulary, Map<String, String> samples, Transformer transformer) throws Exception {
        DataSet<DataRow> dataSet = new DataSet<>();

        for (var entry : samples.entrySet()) {
            String trainInput = entry.getKey();
            String trainOutput = entry.getValue();

            if (!trainOutput.endsWith("<|END|>")) {
                if (!trainOutput.endsWith(" ")) {
                    trainOutput += " ";
                }

                trainOutput += "<|END|>";
            }

            List<String> tokens = vocabulary.split(trainOutput);
            String lastInput = trainInput + " ";

            for (String token : tokens) {
                Tensor input = vocabulary.encode(lastInput);
                Tensor encoded = ENCODING.encode(input);

                Tensor target = TensorFactory.create(vocabulary.getVocabSize());
                int index = vocabulary.wordToIndex(token);

                if (index != -1) {
                    target.set(1, index);
                }

                dataSet.add(new DataRow(encoded, target));
                lastInput += token;
            }
        }

        System.out.println("Fitting with " + dataSet.size() + " samples.");

        long startTime = System.nanoTime();
        transformer.fit(dataSet, 500, 50);
        double duration = (System.nanoTime() - startTime) / 1e6;

        double loss = transformer.loss(dataSet);
        System.out.println("Training took " + duration + " ms with loss " + loss);

        transformer.save("chat_bot");
    }

    private void generateResponse(Vocabulary vocabulary, Transformer model, String prompt) throws InterruptedException {
        StringBuilder botResponse = new StringBuilder();
        String lastWord = "";

        StatesCache cache = new StatesCache(model);

        while (!lastWord.equals("<|END|>") && !lastWord.equals("<|UNK|>")) {
            Tensor input = vocabulary.encode(prompt);
            Tensor encoded = ENCODING.encode(input);
            Tensor output = model.predict(cache, encoded);

            int indexOfMax = BrainUtils.indexOfMaxValue(output);
            String word = vocabulary.indexToWord(indexOfMax);

            botResponse.append(word);
            prompt += word;
            lastWord = word;

            System.out.print("\rChat Bot: " + botResponse);
            Thread.sleep(5);
        }
    }
}
