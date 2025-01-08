package net.echo.brain4j.nlp.attention;

import net.echo.brain4j.utils.Vector;

import java.util.Random;

public class AttentionHead {

    private final int dimension;
    private final double temperature;
    private final double[][] keyWeights; // A key might be a response to the query's question
    private final double[][] queryWeights; // A query is a question for the embedding (ex. "Are adjectives in front of me?")
    private final double[][] valueWeights;

    public AttentionHead(int contextSize, int dimension, double temperature) {
        this.dimension = dimension;
        this.temperature = temperature;

        this.keyWeights = new double[contextSize][dimension];
        this.valueWeights = new double[contextSize][dimension];
        this.queryWeights = new double[contextSize][dimension];

        this.initializeWeights();
    }

    /**
     * Calculates the projection weights of the key, query and values.
     *
     * @param input the input vector or embedding
     * @return the attention vector, or so the required change fpr the embedding.
     */
    public Vector attend(Vector input) {
        double[] key = projectVector(input, keyWeights);
        double[] query = projectVector(input, queryWeights);

        double[] scores = computeScores(query, key);

        double[] values = projectVector(input, valueWeights);
        Vector output = new Vector(values.length);

        for (int i = 0; i < values.length; i++) {
            output.set(i, scores[i] * values[i]);
        }

        return output;
    }

    /**
     * Performs a matrix multiplication.
     */
    public double[] projectVector(Vector embedding, double[][] weights) {
        double[] result = new double[dimension];

        for (int i = 0; i < dimension; i++) {
            double sum = 0;

            for (int j = 0; j < embedding.size(); j++) {
                sum += weights[j][i] * embedding.get(j);
            }

            result[i] = sum;
        }

        return result;
    }

    private void initializeWeights() {
        Random random = new Random();

        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < dimension; j++) {
                queryWeights[i][j] = random.nextGaussian() * 0.02;
                keyWeights[i][j] = random.nextGaussian() * 0.02;
                valueWeights[i][j] = random.nextGaussian() * 0.02;
            }
        }
    }

    private double[] computeScores(double[] query, double[] key) {
        double[] scores = new double[query.length];
        double maxScore = Double.NEGATIVE_INFINITY;

        double queryLength = Math.sqrt(query.length);

        for (int i = 0; i < query.length; i++) {
            scores[i] = (query[i] * key[i]) / queryLength;
            maxScore = Math.max(maxScore, scores[i]);
        }

        double sum = 0.0;

        for (int i = 0; i < scores.length; i++) {
            sum += (scores[i] = Math.exp((scores[i] - maxScore) / temperature));
        }

        for (int i = 0; i < scores.length; i++) {
            scores[i] = scores[i] / sum;
        }

        return scores;
    }
}
