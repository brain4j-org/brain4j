package net.echo.brain4j.transformers.attention;

import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.utils.math.tensor.Tensor;
import net.echo.brain4j.utils.math.tensor.TensorFactory;
import net.echo.brain4j.utils.math.vector.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class AttentionHead {

    protected final int inputDimension;
    protected final int headDimension;
    protected final double temperature;

    protected final float[][] queryWeights;
    protected final float[][] keyWeights;
    protected final float[][] valueWeights;
    
    protected Tensor queryWeightsTensor;
    protected Tensor keyWeightsTensor;
    protected Tensor valueWeightsTensor;

    public AttentionHead(WeightInitializer weightInit, int inputDimension, int headDimension, double temperature) {
        this.inputDimension = inputDimension;
        this.headDimension = headDimension;
        this.temperature = temperature;

        this.queryWeights = new float[inputDimension][headDimension];
        this.keyWeights = new float[inputDimension][headDimension];
        this.valueWeights = new float[inputDimension][headDimension];

        initializeWeights(weightInit);
    }

    public int size() {
        int total = 0;

        total += queryWeights.length * queryWeights[0].length;
        total += keyWeights.length * keyWeights[0].length;
        total += valueWeights.length * valueWeights[0].length;

        return total;
    }
    
    public List<Vector> attendVectors(List<Vector> inputs) {
        int sequenceLength = inputs.size();

        List<Vector> queries = new ArrayList<>();
        List<Vector> keys = new ArrayList<>();
        List<Vector> values = new ArrayList<>();

        for (Vector token : inputs) {
            queries.add(multiply(token, queryWeights));
            keys.add(multiply(token, keyWeights));
            values.add(multiply(token, valueWeights));
        }

        List<Vector> output = new ArrayList<>();
        double scale = Math.sqrt(headDimension);

        for (int i = 0; i < sequenceLength; i++) {
            Vector query = queries.get(i);
            List<Double> scoreList = new ArrayList<>();

            for (int j = 0; j < sequenceLength; j++) {
                double score = query.weightedSum(keys.get(j)) / scale;
                scoreList.add(score);
            }

            Vector attentionWeights = softmax(scoreList);
            Vector headOutput = new Vector(headDimension);

            for (int j = 0; j < sequenceLength; j++) {
                Vector weightedValue = values.get(j).scale(attentionWeights.get(j));
                headOutput = headOutput.add(weightedValue);
            }

            output.add(headOutput);
        }

        return output;
    }
    
    public List<Vector> attend(List<Vector> inputs) {
        return attendVectors(inputs);
    }
    
    public List<Tensor> attendTensors(List<Tensor> inputs) {
        int sequenceLength = inputs.size();

        ensureWeightTensors();

        List<Tensor> queries = new ArrayList<>();
        List<Tensor> keys = new ArrayList<>();
        List<Tensor> values = new ArrayList<>();

        for (Tensor token : inputs) {
            queries.add(token.matmul(queryWeightsTensor));
            keys.add(token.matmul(keyWeightsTensor));
            values.add(token.matmul(valueWeightsTensor));
        }

        List<Tensor> output = new ArrayList<>();
        double scale = Math.sqrt(headDimension);

        for (int i = 0; i < sequenceLength; i++) {
            Tensor query = queries.get(i);
            List<Double> scoreList = new ArrayList<>();

            for (int j = 0; j < sequenceLength; j++) {
                double score = query.dot(keys.get(j)) / scale;
                scoreList.add(score);
            }

            Vector attentionWeightsVec = softmax(scoreList);
            Tensor attentionWeights = TensorFactory.vector(attentionWeightsVec);
            
            Tensor headOutput = TensorFactory.zeros(headDimension);

            for (int j = 0; j < sequenceLength; j++) {
                Tensor weightedValue = values.get(j).mul(attentionWeights.get(j));
                headOutput = headOutput.add(weightedValue);
            }

            output.add(headOutput);
        }

        return output;
    }

    protected void initializeWeights(WeightInitializer initializer) {
        Random rng = new Random();

        double bound = initializer.getBound(inputDimension, headDimension);

        for (int i = 0; i < inputDimension; i++) {
            for (int j = 0; j < headDimension; j++) {
                queryWeights[i][j] = (float) (rng.nextDouble(2 * bound) - bound);
                keyWeights[i][j] = (float) (rng.nextDouble(2 * bound) - bound);
                valueWeights[i][j] = (float) (rng.nextDouble(2 * bound) - bound);
            }
        }
    }
    
    protected void ensureWeightTensors() {
        if (queryWeightsTensor == null) {
            float[] qWeights = matrixToArray(queryWeights);
            float[] kWeights = matrixToArray(keyWeights);
            float[] vWeights = matrixToArray(valueWeights);
            
            queryWeightsTensor = TensorFactory.of(new int[]{inputDimension, headDimension}, qWeights);
            keyWeightsTensor = TensorFactory.of(new int[]{inputDimension, headDimension}, kWeights);
            valueWeightsTensor = TensorFactory.of(new int[]{inputDimension, headDimension}, vWeights);
        }
    }
    
    private float[] matrixToArray(float[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[] result = new float[rows * cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[i * cols + j] = matrix[i][j];
            }
        }
        
        return result;
    }

    protected Vector multiply(Vector vector, float[][] weights) {
        Vector result = new Vector(headDimension);

        for (int j = 0; j < headDimension; j++) {
            double sum = 0.0;

            for (int i = 0; i < inputDimension; i++) {
                sum += vector.get(i) * weights[i][j];
            }

            result.set(j, sum);
        }
        return result;
    }

    protected Vector softmax(List<Double> scores) {
        Vector result = new Vector(scores.size());
        double maxScore = Double.NEGATIVE_INFINITY;

        for (double score : scores) {
            if (score > maxScore) {
                maxScore = score;
            }
        }

        double sum = 0.0;

        for (int i = 0; i < scores.size(); i++) {
            double expVal = Math.exp((scores.get(i) - maxScore) / temperature);
            result.set(i, expVal);

            sum += expVal;
        }

        for (int i = 0; i < result.size(); i++) {
            result.set(i, result.get(i) / sum);
        }

        return result;
    }
}
