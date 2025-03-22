package net.echo.brain4j.transformers.masked;

import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;
import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.transformers.attention.AttentionHead;
import net.echo.math4j.math.vector.Vector;

import java.util.ArrayList;
import java.util.List;

public class MaskedAttentionHead extends AttentionHead {

    public MaskedAttentionHead(WeightInitializer weightInit, int inputDimension, int headDimension, double temperature) {
        super(weightInit, inputDimension, headDimension, temperature);
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

            for (int j = 0; j <= i; j++) {
                double score = query.dot(keys.get(j)) / scale;
                scoreList.add(score);
            }

            Vector attentionWeightsVec = softmax(scoreList);
            Tensor attentionWeights = TensorFactory.vector(attentionWeightsVec);
            
            Tensor headOutput = TensorFactory.zeros(headDimension);

            for (int j = 0; j <= i; j++) {
                float attention = attentionWeights.get(j);

                Tensor valueTensor = values.get(j);
                Tensor weightedValue = valueTensor.mul(attention);

                headOutput = headOutput.add(weightedValue);
            }

            output.add(headOutput);
        }

        return output;
    }
}
