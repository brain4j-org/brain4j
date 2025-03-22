package net.echo.brain4j.transformers.attention;

import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.math4j.math.vector.Vector;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class AttentionHead {

    protected final int inputDimension;
    protected final int headDimension;
    protected final double temperature;

    protected final Tensor queryWeightsTensor;
    protected final Tensor keyWeightsTensor;
    protected final Tensor valueWeightsTensor;

    public AttentionHead(WeightInitializer weightInit, int inputDimension, int headDimension, double temperature) {
        this.inputDimension = inputDimension;
        this.headDimension = headDimension;
        this.temperature = temperature;

        this.queryWeightsTensor = TensorFactory.matrix(inputDimension, headDimension);
        this.keyWeightsTensor = TensorFactory.matrix(inputDimension, headDimension);
        this.valueWeightsTensor = TensorFactory.matrix(inputDimension, headDimension);

        initializeWeights(weightInit);
    }

    public int size() {
        return 3 * inputDimension * headDimension;
    }

    public Tensor attend(Tensor input) {
        Tensor Q = input.matmul(queryWeightsTensor);
        Tensor K = input.matmul(keyWeightsTensor);
        Tensor V = input.matmul(valueWeightsTensor);

        double normalizer = Math.sqrt(headDimension);

        Tensor scores = Q.matmul(K.transpose()).div(normalizer);
        Tensor attentionWeights = scores.softmax();

        return attentionWeights.matmul(V);
    }
    
    public List<Tensor> attendTensors(List<Tensor> inputs) {
        int sequenceLength = inputs.size();

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
            
            Tensor headOutput = TensorFactory.zeros(1, headDimension);

            for (int j = 0; j < sequenceLength; j++) {
                float attention = attentionWeights.get(j);

                Tensor valueTensor = values.get(j);
                Tensor weightedValue = valueTensor.mul(attention);

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
                queryWeightsTensor.set(rng.nextDouble(2 * bound) - bound, i, j);
                keyWeightsTensor.set(rng.nextDouble(2 * bound) - bound, i, j);
                valueWeightsTensor.set(rng.nextDouble(2 * bound) - bound, i, j);
            }
        }
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
