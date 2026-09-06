package org.brain4j.core.training.wrappers;

import org.brain4j.math.Tensors;
import org.brain4j.math.commons.Commons;
import org.brain4j.math.tensor.Tensor;

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
    private double totalDeviation;

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
        this.totalDeviation = calculateDeviation();
    }

    private double calculateDeviation() {
        double sum = 0.0;
        double total = 0.0;
        int count = 0;

        for (Tensor tensor : classifications.values()) {
            for (int i = 0; i < tensor.elements(); i++) {
                total += tensor.get(i);
                count++;
            }
        }

        if (count == 0) return 0.0;

        double mean = total / count;

        for (Tensor tensor : classifications.values()) {
            for (int i = 0; i < tensor.elements(); i++) {
                double diff = tensor.get(i) - mean;
                sum += diff * diff;
            }
        }

        return sum;
    }

    public String results() {
        StringBuilder matrix = new StringBuilder();
        String divider = Commons.getHeader(" Evaluation Results ", Commons.HEADER_CHAR);

        matrix.append(divider);
        matrix.append("Out of ").append(classifications.size()).append(" classes\n\n");

        String secondary = "%-12s %-10s\n";
        matrix.append(secondary.formatted("Loss:", "%.4f".formatted(loss)));
        matrix.append(secondary.formatted("Accuracy:", "%.4f".formatted(accuracy)));
        matrix.append(secondary.formatted("Precision:", "%.4f".formatted(precision)));
        matrix.append(secondary.formatted("Recall:", "%.4f".formatted(recall)));
        matrix.append(secondary.formatted("F1-score:", "%.4f".formatted(f1Score)));
        
        if (!classifications.isEmpty()) {
            divider = Commons.getHeader(" Confusion Matrix ", Commons.HEADER_CHAR);
            matrix.append(divider);
            matrix.append("First column is the actual class, top row are the predicted classes.\n\n");
            
            int maxValue = 0;
            for (Tensor tensor : classifications.values()) {
                for (int i = 0; i < tensor.elements(); i++) {
                    maxValue = Math.max(maxValue, (int) tensor.get(i));
                }
            }
            
            int cellWidth = Math.max(String.valueOf(maxValue).length(), 5) + 1;
            
            matrix.repeat(" ", 7);
            for (int i = 0; i < classes; i++) {
                matrix.append(("%" + cellWidth + "d").formatted(i));
            }
            
            matrix.append("\n  ");
            matrix.repeat("-", 6 + cellWidth * classes).append("\n");

            for (int i = 0; i < classes; i++) {
                matrix.append("%4d | ".formatted(i));
                Tensor predictions = classifications.get(i);
                
                for (int j = 0; j < predictions.elements(); j++) {
                    int prediction = (int) predictions.get(j);
                    matrix.append(("%" + cellWidth + "d").formatted(prediction));
                }
                matrix.append("\n");
            }
            
            matrix.append("\n");
        }

        matrix.append(Commons.getHeader("", Commons.HEADER_CHAR));

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

    public Map<Integer, Tensor> classifications() {
        return classifications;
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

    public double totalDeviation() {
        return totalDeviation;
    }
}
