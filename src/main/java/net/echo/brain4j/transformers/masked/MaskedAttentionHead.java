package net.echo.brain4j.transformers.masked;

import net.echo.brain4j.model.initialization.WeightInit;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.transformers.attention.AttentionHead;
import net.echo.brain4j.utils.math.tensor.Tensor;
import net.echo.brain4j.utils.math.tensor.TensorFactory;
import net.echo.brain4j.utils.math.vector.Vector;

import java.util.ArrayList;
import java.util.List;

public class MaskedAttentionHead extends AttentionHead {

    public MaskedAttentionHead(WeightInitializer weightInit, int inputDimension, int headDimension, double temperature) {
        super(weightInit, inputDimension, headDimension, temperature);
    }

    @Override
    public List<Vector> attend(List<Vector> inputs) {
        return attendVectors(inputs);
    }
    
    @Override
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

            for (int j = 0; j <= i; j++) {
                double score = query.weightedSum(keys.get(j)) / scale;
                scoreList.add(score);
            }

            Vector attentionWeights = softmax(scoreList);
            Vector headOutput = new Vector(headDimension);

            for (int j = 0; j <= i; j++) {
                headOutput = headOutput.add(values.get(j).scale(attentionWeights.get(j)));
            }

            output.add(headOutput);
        }

        return output;
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

            for (int j = 0; j <= i; j++) {
                double score = query.dot(keys.get(j)) / scale;
                scoreList.add(score);
            }

            Vector attentionWeightsVec = softmax(scoreList);
            Tensor attentionWeights = TensorFactory.vector(attentionWeightsVec);
            
            Tensor headOutput = TensorFactory.zeros(headDimension);

            for (int j = 0; j <= i; j++) {
                Tensor weightedValue = values.get(j).mul(attentionWeights.get(j));
                headOutput = headOutput.add(weightedValue);
            }

            output.add(headOutput);
        }

        return output;
    }
}
