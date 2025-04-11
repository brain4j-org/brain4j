package gpt;

import net.echo.brain4j.Brain4J;
import net.echo.brain4j.loss.Loss;
import net.echo.brain4j.model.impl.Transformer;
import net.echo.brain4j.structure.StatesCache;
import net.echo.brain4j.training.advanced.SmartTrainer;
import net.echo.brain4j.training.advanced.TrainListener;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.evaluation.EvaluationResult;
import net.echo.brain4j.training.optimizer.impl.Adam;
import net.echo.brain4j.transformers.ContextWindow;
import net.echo.brain4j.transformers.TransformerDecoder;
import net.echo.brain4j.transformers.encoding.PositionalEncoding;
import net.echo.brain4j.transformers.vocabulary.Vocabulary;
import net.echo.brain4j.transformers.vocabulary.VocabularyMapper;
import net.echo.math4j.BrainUtils;
import net.echo.math4j.DataSet;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class ChatBotExample {

    private static final int EMBEDDING_SIZE = 64;
    private static final PositionalEncoding ENCODING = new PositionalEncoding(200, EMBEDDING_SIZE);

    public static void main(String[] args) throws Exception {
        new ChatBotExample().start();
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

        Map<String, String> samples = Map.of(
            "write a story", "Once upon a time, there was a small cat named Mia. Mia lived in a cozy house with Sarah. Every day, Mia played in the garden and chased butterflies. One day, she found a shiny key. It opened a hidden room full of toys. Mia was very happy!"
        );

        DataSet<DataRow> dataSet = getDataSet(vocabulary, samples);
        Scanner scanner = new Scanner(System.in);

        System.out.print("Have you got a pre-trained model? (y/n): ");
        String shouldLoad = scanner.nextLine();

        if (shouldLoad.equals("y")) {
            System.out.print("What's the model name? ");
            String name = scanner.nextLine();

            model.load(name + (!name.endsWith(".b4j") ? ".b4j" : ""));
        }

        System.out.println(model.summary());

        System.out.print("Train model? (y/n): ");
        String shouldTrain = scanner.nextLine();

        if (shouldTrain.equals("y")) {
            trainModel(dataSet, model);

            System.out.print("Insert a name for the model: ");

            String name = scanner.nextLine();
            String path = name + (!name.endsWith("chatbot.b4j") ? "chatbot.b4j" : "");

            model.save(path);
            System.out.printf("Saved model inside %s!%n", path);
        }

        System.out.print("Would you like to use the GPU? (y/n): ");
        String shouldUseGPU = scanner.nextLine();

        if (shouldUseGPU.equals("y")) {
            Brain4J.useGPUIfAvailable();
        }

        EvaluationResult result = model.evaluate(dataSet);
        System.out.printf("Accuracy: %.3f%% Loss: %.3f%n", result.accuracy() * 100, result.loss());

        inference(vocabulary, model, scanner);
    }

    private void inference(Vocabulary vocabulary, Transformer model, Scanner scanner) {
        System.out.print("Enter a prompt (Enter 'end' to exit): ");
        String prompt = scanner.nextLine() + " ";

        if (prompt.startsWith("end")) {
            return;
        }

        generateResponse(vocabulary, model, prompt);
        System.out.println();

        inference(vocabulary, model, scanner);
    }

    private DataSet<DataRow> getDataSet(Vocabulary vocabulary, Map<String, String> samples) {
        DataSet<DataRow> dataSet = new DataSet<>();
        Map<String, Tensor> inputEncodingCache = new HashMap<>();

        for (Map.Entry<String, String> entry : samples.entrySet()) {
            String trainInput = entry.getKey();
            String trainOutput = entry.getValue();

            if (!trainOutput.endsWith("<END>")) {
                if (!trainOutput.endsWith(" ")) {
                    trainOutput += " ";
                }

                trainOutput += "<END>";
            }

            List<String> tokens = vocabulary.split(trainOutput);
            String lastInput = trainInput + " ";

            for (String token : tokens) {
                Tensor encoded;

                if (inputEncodingCache.containsKey(lastInput)) {
                    encoded = inputEncodingCache.get(lastInput);
                } else {
                    Tensor input = vocabulary.encode(lastInput);
                    encoded = ENCODING.encode(input);

                    if (lastInput.length() < 50) {
                        inputEncodingCache.put(lastInput, encoded);
                    }
                }

                Tensor target = TensorFactory.create(vocabulary.getVocabSize());
                int index = vocabulary.wordToIndex(token);

                if (index != -1) {
                    target.set(1, index);
                }

                dataSet.add(new DataRow(encoded, target));
                lastInput += token;
            }
        }

        return dataSet;
    }

    private void trainModel(DataSet<DataRow> dataSet, Transformer model) {
        System.out.println("Fitting with " + dataSet.size() + " samples.");

        long startTime = System.nanoTime();

        SmartTrainer trainer = new SmartTrainer(0.01, 50);

        trainer.addListener(new Evaluator());
        trainer.startFor(model, dataSet, 500);

        double duration = (System.nanoTime() - startTime) / 1e6;
        System.out.println("Training took " + duration + " ms");
    }

    private void generateResponse(Vocabulary vocabulary, Transformer model, String prompt) {
        ContextWindow window = new ContextWindow(256, EMBEDDING_SIZE);
        Tensor start = vocabulary.encode(prompt);

        window.setContext(start);

        StringBuilder botResponse = new StringBuilder();
        StatesCache cache = new StatesCache();

        String modeCollapse = " ".repeat(10);
        double totalTime = 0.0;

        while (true) {
            long startTime = System.nanoTime();

            Tensor context = ENCODING.encode(window.toInput());
            Tensor output = model.predict(cache, context);

            double took = (System.nanoTime() - startTime) / 1e6;

            int argmax = BrainUtils.argmax(output);
            String word = vocabulary.indexToWord(argmax);
            Tensor encoded = vocabulary.wordToVec(word);

            window.append(encoded);
            botResponse.append(word);
            totalTime += took;

            if (word.equals("<END>") || word.equals("<UNK>") || botResponse.toString().endsWith(modeCollapse)) {
                break;
            }

            System.out.printf("\r%.2fms/token - %s", took, botResponse);
        }

        System.out.printf("\nTotal time: %.3f ms", totalTime);
    }

    static class Evaluator extends TrainListener {

        @Override
        public void onEpochCompleted(int epoch, int totalEpoches, long took) {
            if (totalEpoches == Integer.MAX_VALUE) return;

            model.printProgressBar(epoch + 1, totalEpoches, trainer.getEvaluateEvery());
        }

        @Override
        public void onEvaluated(DataSet<DataRow> dataSet, EvaluationResult evaluation, int epoch, long took) {
            if (evaluation.accuracy() == 1 && evaluation.loss() < 0.2) {
                trainer.abort();
            }

            System.out.printf("Loss at epoch %s: %.4f | Accuracy: %.2f%%\n", epoch, evaluation.loss(),
                    evaluation.accuracy() * 100);
        }
    }
}
