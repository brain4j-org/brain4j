package net.echo.brain4j.training.evaluation;

import java.util.Map;

public record EvaluationResult(int classes, Map<Integer, Integer> correctlyClassified, Map<Integer, Integer> incorrectlyClassified) {

    public String confusionMatrix() {
        StringBuilder matrix = new StringBuilder();
        String header = "======================================\n";

        String pattern = "%-10s %-10s %-10s\n";
        matrix.append(String.format(pattern, "Classes", "Correct", "Incorrect"));
        matrix.append(header);

        int totalCorrect = 0;
        int totalIncorrect = 0;

        for (int i = 0; i < correctlyClassified.size(); i++) {
            totalCorrect += correctlyClassified.get(i);
            totalIncorrect += incorrectlyClassified.get(i);
        }

        for (int i = 0; i < classes; i++) {
            matrix.append(String.format(pattern, i, correctlyClassified.get(i), incorrectlyClassified.get(i)));
        }

        matrix.append(header);

        double accuracy = totalCorrect / (double) (totalCorrect + totalIncorrect);
        double precision = totalCorrect / (double) (totalCorrect + incorrectlyClassified.get(0));
        double recall = totalCorrect / (double) (totalCorrect + correctlyClassified.get(0));

        String secondary = "%-20s %-10s\n";
        matrix.append(String.format(secondary, "Accuracy:", String.format("%.2f", accuracy * 100), ""));
        matrix.append(String.format(secondary, "Precision:", String.format("%.2f", precision * 100), ""));
        matrix.append(String.format(secondary, "Recall: ", String.format("%.2f", recall * 100), ""));
        matrix.append(header);

        return matrix.toString();
    }
}
