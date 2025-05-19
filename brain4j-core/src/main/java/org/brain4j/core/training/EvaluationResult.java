package org.brain4j.core.training;

import org.brain4j.math.Commons;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;

import java.util.Map;

public class EvaluationResult {

    private final double loss;
    private final int classes;
    private final Map<Integer, Tensor> classifications;

    private int totalCorrect;
    private int totalIncorrect;
    private double accuracy;
    private double precision;
    private double recall;
    private double f1Score;

    public EvaluationResult(double loss, int classes, Map<Integer, Tensor> classifications) {
        this.loss = loss;
        this.classes = classes;
        this.classifications = classifications;
        calculateStats();
    }

    private void calculateStats() {
        int[] truePositives = new int[classes];
        int[] falsePositives = new int[classes];
        int[] falseNegatives = new int[classes];

        for (int i = 0; i < classifications.size(); i++) {
            Tensor vector = classifications.get(i);

            for (int j = 0; j < vector.elements(); j++) {
                int value = (int) vector.get(j);

                if (i == j) {
                    totalCorrect += value;
                    truePositives[i] += value;
                } else {
                    totalIncorrect += value;
                    falsePositives[j] += value;
                    falseNegatives[i] += value;
                }
            }
        }

        double precisionSum = 0, recallSum = 0;
        for (int i = 0; i < classes; i++) {
            double precision = (truePositives[i] + falsePositives[i]) > 0 ?
                    (double) truePositives[i] / (truePositives[i] + falsePositives[i]) : 0;

            double recall = (truePositives[i] + falseNegatives[i]) > 0 ?
                    (double) truePositives[i] / (truePositives[i] + falseNegatives[i]) : 0;

            precisionSum += precision;
            recallSum += recall;
        }

        this.accuracy = (double) totalCorrect / (totalCorrect + totalIncorrect);
        this.precision = precisionSum / classes;
        this.recall = recallSum / classes;
        this.f1Score = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
    }

    public String results() {
        StringBuilder matrix = new StringBuilder();
        String divider = Commons.getHeader(" Evaluation Results ", Commons.getHeaderChar());

        matrix.append(divider);
        matrix.append("Out of ").append(classifications.size()).append(" classes\n\n");

        String secondary = "%-12s %-10s\n";
        matrix.append(secondary.formatted("Loss:", "%.4f".formatted(loss)));
        matrix.append(secondary.formatted("Accuracy:", "%.4f".formatted(accuracy)));
        matrix.append(secondary.formatted("Precision:", "%.4f".formatted(precision)));
        matrix.append(secondary.formatted("Recall:", "%.4f".formatted(recall)));
        matrix.append(secondary.formatted("F1-score:", "%.4f".formatted(f1Score)));

        if (!classifications.isEmpty()) {
            divider = Commons.getHeader(" Confusion Matrix ", Commons.getHeaderChar());
            matrix.append(divider);
            matrix.append("First column is the actual class, top row are the predicted classes.\n\n");
            matrix.append(" ".repeat(7));

            for (int i = 0; i < classes; i++) {
                matrix.append("%4d".formatted(i)).append(" ");
            }

            matrix.append("\n  ");
            matrix.append("-".repeat(5 + classes * 5)).append("\n");

            for (int i = 0; i < classes; i++) {
                StringBuilder text = new StringBuilder();
                Tensor predictions = classifications.get(i);

                for (int j = 0; j < predictions.elements(); j++) {
                    int prediction = (int) predictions.get(j);
                    text.append("%4d".formatted(prediction)).append(" ");
                }

                matrix.append("%4d | ".formatted(i));
                matrix.append(text).append("\n");
            }

            matrix.append("\n");
        }

        matrix.append(Commons.getHeader("", Commons.getHeaderChar()));

        return matrix.toString();
    }

    public Tensor confusionMatrix() {
        Tensor result = Tensors.matrix(classes, classes);

        for (int i = 0; i < classes; i++) {
            Tensor predictions = classifications.get(i);

            for (int j = 0; j < predictions.elements(); j++) {
                result.set(predictions.get(j), i, j);
            }
        }

        return result;
    }

    public double loss() {
        return loss;
    }

    public int classes() {
        return classes;
    }

    public int totalCorrect() {
        return totalCorrect;
    }

    public int totalIncorrect() {
        return totalIncorrect;
    }

    public double accuracy() {
        return accuracy;
    }

    public double precision() {
        return precision;
    }

    public double recall() {
        return recall;
    }

    public double f1Score() {
        return f1Score;
    }

    public Map<Integer, Tensor> classifications() {
        return classifications;
    }
}
