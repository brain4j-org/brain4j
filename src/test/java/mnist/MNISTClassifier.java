package mnist;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.impl.Sequential;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.training.techniques.SmartTrainer;
import net.echo.brain4j.training.techniques.TrainListener;
import net.echo.brain4j.utils.DataSet;
import net.echo.brain4j.utils.MLUtils;
import net.echo.brain4j.utils.Vector;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class MNISTClassifier {

    public static void main(String[] args) {
        DataSet<DataRow> set = getData();
        Sequential model = getModel();

        System.out.println(model.getStats());
        set.partition(32);

        double loss = model.evaluate(set);
        System.out.println("Initial loss: " + loss);

        train(model, set);
        evaluateModel(model);
    }

    public static void evaluateModel(Sequential model) {
        model.reloadMatrices();

        DataSet<DataRow> set = getData();
        Map<Integer, Vector> results = initializeResults(set);

        int correctlyClassified = 0;
        int incorrectlyClassified = 0;

        for (DataRow row : set) {
            Vector output = model.predict(row.inputs());

            int prediction = MLUtils.indexOfMaxValue(output);
            int expected = MLUtils.indexOfMaxValue(row.outputs());

            Vector result = results.get(expected);

            if (prediction == expected) {
                correctlyClassified++;
                result.set(0, result.get(0) + 1);
            } else {
                incorrectlyClassified++;
                result.set(1, result.get(1) + 1);
            }
        }

        printEvaluationResults(results, model.evaluate(set), correctlyClassified, incorrectlyClassified);
    }

    private static Map<Integer, Vector> initializeResults(DataSet<DataRow> set) {
        Map<Integer, Vector> results = new HashMap<>();

        for (DataRow row : set.getData()) {
            int number = MLUtils.indexOfMaxValue(row.outputs());
            results.put(number, new Vector(2));
        }

        return results;
    }

    private static void printEvaluationResults(Map<Integer, Vector> results, double loss, int correctlyClassified, int incorrectlyClassified) {
        for (Map.Entry<Integer, Vector> entry : results.entrySet()) {
            System.out.println("Label: " + entry.getKey() + " -> (Correct, Incorrect) " + entry.getValue());
        }

        double accuracy = (double) correctlyClassified / (correctlyClassified + incorrectlyClassified);

        System.out.println("----------- Evaluation Results -----------");
        System.out.println("Correctly classified: " + correctlyClassified);
        System.out.println("Incorrectly classified: " + incorrectlyClassified);
        System.out.printf("Accuracy: %.4f (%.4f%%)%n", accuracy, accuracy * 100);
        System.out.printf("Loss: %.6f\n", loss);
        System.out.println("----------- Finished Evaluation -----------");
    }

    public static void train(Model model, DataSet<DataRow> set) {
        SmartTrainer trainer = new SmartTrainer(0.7, 10);

        trainer.addListener(new ExampleListener());
        trainer.startFor(model, set, 150, 0.01);

        double loss = model.evaluate(set);
        model.save("mnist_" + loss + ".json");

        System.out.println("Took: " + trainer.getTook() / 1e6);
    }

    public static Sequential getModel() {
        Sequential model = new Sequential(
                new DenseLayer(784, Activations.LINEAR),
                new DenseLayer(64, Activations.SIGMOID),
                new DenseLayer(10, Activations.SOFTMAX)
        );

        return model.compile(LossFunctions.CROSS_ENTROPY, new Adam(0.01));
    }

    public static DataSet<DataRow> getData() {
        DataSet<DataRow> set = new DataSet<>();

        try {
            List<String> lines = FileUtils.readLines(new File("dataset.txt"), "UTF-8");

            for (String line : lines) {
                String[] parts = line.split(",");
                double[] inputs = Arrays.stream(parts, 1, parts.length).mapToDouble(Double::parseDouble).toArray();

                Vector output = new Vector(10);

                int value = Integer.parseInt(parts[0]);
                output.set(value, 1);

                set.getData().add(new DataRow(Vector.of(inputs), output));
            }
        } catch (IOException e) {
            throw new RuntimeException("Error reading dataset: " + e.getMessage(), e);
        }

        return set;
    }

    private static class ExampleListener extends TrainListener<DataRow> {

        @Override
        public void onEvaluated(DataSet<DataRow> dataSet, int epoch, double loss, long took) {
            System.out.print("\rEpoch " + epoch + " loss: " + loss + " took " + (took / 1e6) + " ms");
        }
    }
}