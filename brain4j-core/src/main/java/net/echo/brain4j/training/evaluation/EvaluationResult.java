package net.echo.brain4j.training.evaluation;

import net.echo.math4j.BrainUtils;
import net.echo.math4j.math.tensor.Tensor;

import java.util.Map;

public record EvaluationResult(int classes, Map<Integer, Tensor> classifications) {

    public String confusionMatrix() {
        StringBuilder matrix = new StringBuilder();
        String divider = BrainUtils.getHeader(" Evaluation Results ");

        matrix.append(divider);
        matrix.append("Out of ").append(classifications.size()).append(" classes\n\n");

        int totalCorrect = 0;
        int totalIncorrect = 0;

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

        // Accuracy
        double accuracy = (double) totalCorrect / (totalCorrect + totalIncorrect);

        double precisionSum = 0, recallSum = 0;
        for (int i = 0; i < classes; i++) {
            double precision = (truePositives[i] + falsePositives[i]) > 0 ?
                    (double) truePositives[i] / (truePositives[i] + falsePositives[i]) : 0;

            double recall = (truePositives[i] + falseNegatives[i]) > 0 ?
                    (double) truePositives[i] / (truePositives[i] + falseNegatives[i]) : 0;

            precisionSum += precision;
            recallSum += recall;
        }

        double precision = precisionSum / classes;
        double recall = recallSum / classes;
        double f1Score = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;

        String secondary = "%-20s %-10s\n";
        matrix.append(String.format(secondary, "Accuracy:", String.format("%.4f", accuracy)));
        matrix.append(String.format(secondary, "Precision:", String.format("%.4f", precision)));
        matrix.append(String.format(secondary, "Recall:", String.format("%.4f", recall)));
        matrix.append(String.format(secondary, "F1-score:", String.format("%.4f", f1Score)));

        divider = BrainUtils.getHeader(" Confusion Matrix ");
        matrix.append(divider);
        matrix.append("First column is the actual class, top row are the predicted classes.\n\n");
        matrix.append("       ");

        for (int i = 0; i < classes; i++) {
            matrix.append(String.format("%4d", i)).append(" ");
        }

        matrix.append("\n  ");
        matrix.append("-".repeat(5 + classes * 5)).append("\n");

        for (int i = 0; i < classes; i++) {
            StringBuilder text = new StringBuilder();
            Tensor predictions = classifications.get(i);

            for (int j = 0; j < predictions.elements(); j++) {
                text.append(String.format("%4d", (int) predictions.get(j))).append(" ");
            }

            matrix.append(String.format("%4d | ", i));
            matrix.append(text).append("\n");
        }

        matrix.append("\n");
        matrix.append("=".repeat(divider.length() - 1));
        return matrix.toString();
    }
}
