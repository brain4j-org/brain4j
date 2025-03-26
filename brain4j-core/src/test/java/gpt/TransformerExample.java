package gpt;

import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.impl.Transformer;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.optimizer.impl.Adam;
import net.echo.brain4j.transformers.TransformerEncoder;
import net.echo.brain4j.transformers.vocabulary.Vocabulary;
import net.echo.brain4j.transformers.vocabulary.VocabularyMapper;
import net.echo.brain4j.transformers.encoding.PositionalEncoding;
import net.echo.math4j.BrainUtils;
import net.echo.math4j.DataSet;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class TransformerExample {

    private static final int EMBEDDING_SIZE = 16;
    private static final PositionalEncoding ENCODING = new PositionalEncoding(100, EMBEDDING_SIZE);

    public static void main(String[] args) throws Exception {
        new TransformerExample().start();
    }

    private List<String> loadExamples() throws IOException {
        return Files.readAllLines(Path.of("dataset.txt"));
    }

    private void start() throws Exception {
        List<String> examples = loadExamples();

        Vocabulary vocabulary = new Vocabulary(examples, EMBEDDING_SIZE);
        vocabulary.tokenize();

        Transformer model = new Transformer(
                new TransformerEncoder(4, EMBEDDING_SIZE),
                new VocabularyMapper(vocabulary.getVocabSize(), EMBEDDING_SIZE, 0.1)
        );

        model.compile(LossFunctions.CROSS_ENTROPY, new Adam(0.1));

        System.out.println(model.summary());
        System.out.println("Vocabulary size: " + vocabulary.getVocabSize());

        Scanner scanner = new Scanner(System.in);

        Map<String, String> samples = Map.of(
                "nice guy", "thank you!",
                "what is your name", "my name is brain4j",
                "hi", "hey, what is up?"
                //  "write a story", "Hi! Once upon a time, there was a fox. It was walking in the forest, thinking about the day. The sky was clear, and the moon was shining bright. It looked up and saw something strange. A star fell. The fox was curious and went closer. It touched the ground, and the star was gone. But the fox smiled, knowing that magic happens when least expected.<END>"
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

    private void trainModel(Vocabulary vocabulary, Map<String, String> samples, Transformer transformer) {
        DataSet<DataRow> dataSet = new DataSet<>();

        for (var entry : samples.entrySet()) {
            String trainInput = entry.getKey();
            String trainOutput = entry.getValue();

            if (!trainOutput.endsWith("<END>")) {
                trainOutput += "<END>";
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
        for (int i = 0; i < 100; i++) {
            transformer.fit(dataSet, 100);

            double loss = transformer.loss(dataSet);
            System.out.println("Loss " + loss);
        }
        double duration = (System.nanoTime() - startTime) / 1e6;

        double loss = transformer.loss(dataSet);
        System.out.println("Training took " + duration + " ms with loss " + loss);
    }

    private void generateResponse(Vocabulary vocabulary, Transformer transformer, String prompt) throws InterruptedException {
        StringBuilder botResponse = new StringBuilder();
        String lastWord = "";

        while (!lastWord.equals("<END>")) {
            Tensor input = vocabulary.encode(prompt);
            Tensor encoded = ENCODING.encode(input);
            Tensor output = transformer.predict(encoded);

            int indexOfMax = BrainUtils.indexOfMaxValue(output);
            String word = vocabulary.indexToWord(indexOfMax);

            botResponse.append(word);
            prompt += word;
            lastWord = word;

            System.out.print("\rChat Bot: " + botResponse);
            Thread.sleep(50);
        }
    }
}
