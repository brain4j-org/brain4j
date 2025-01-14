package mnist;

import net.echo.brain4j.activation.Activations;
import net.echo.brain4j.layer.impl.DenseLayer;
import net.echo.brain4j.loss.LossFunctions;
import net.echo.brain4j.model.Model;
import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.training.data.DataRow;
import net.echo.brain4j.training.data.DataSet;
import net.echo.brain4j.training.optimizers.impl.Adam;
import net.echo.brain4j.training.updater.impl.NormalUpdater;
import net.echo.brain4j.utils.MLUtils;
import net.echo.brain4j.utils.Vector;
import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class MNISTClassifier {

    public static void main(String[] args) {
        DataSet set = getData();
        Model model = getModel();

        set.partition(32);
        evaluateModel(model);
    }

    public static void evaluateModel(Model model) {
        model.load("mnist-56.json");
        model.reloadMatrices();

        DataSet set = getData();
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

                System.out.println(expected + " is correct. Vector: " + output.toString("%.3f"));
            } else {
                incorrectlyClassified++;
                result.set(1, result.get(1) + 1);

                System.out.println(expected + " is incorrect. Vector: " + output.toString("%.3f"));
            }
        }

        printEvaluationResults(results, model.evaluate(set), correctlyClassified, incorrectlyClassified);
    }

    private static Map<Integer, Vector> initializeResults(DataSet set) {
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
        System.out.printf("Loss: %.6f", loss);
    }

    public static void train(Model model, DataSet set) {
        for (int i = 0; i < 1000; i++) {
            long start = System.nanoTime();
            model.fit(set);

            if (i % 10 == 0) {
                double took = (System.nanoTime() - start) / 1e6;
                double error = model.evaluate(set);

                System.out.println("Epoch " + i + " took " + took + " ms and has " + error + " loss");
            }

            if (i % 100 == 0) {
                model.save("mnist-56.json");
            }
        }
    }

    public static Model getModel() {
        Model model = new Model(
                new DenseLayer(784, Activations.LINEAR),
                new DenseLayer(32, Activations.RELU),
                new DenseLayer(10, Activations.SOFTMAX)
        );

        model.compile(
                WeightInit.UNIFORM_XAVIER,
                LossFunctions.CATEGORICAL_CROSS_ENTROPY,
                new Adam(0.001),
                new NormalUpdater()
        );

        return model;
    }

    public static DataSet getData() {
        DataSet set = new DataSet();

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
}