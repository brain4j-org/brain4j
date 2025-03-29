package net.echo.brain4j.transformers.head;

import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;
import net.echo.math4j.math.tensor.index.Range;

import java.util.List;

public class MaskedAttentionHead extends AttentionHead {

    public MaskedAttentionHead(int inputDimension, int headDimension) {
        super(inputDimension, headDimension);
    }

    @Override
    public Tensor attend(Tensor input) {
        Tensor Q = input.matmul(queryWeightsTensor);
        
        int seqLength = input.shape()[0];
        int cachedLength = keyCache.size();
        
        Tensor K, V;
        if (isUsingCache() && cachedLength > 0) {
            if (seqLength > cachedLength) {
                Range range = new Range(cachedLength, seqLength);
                Tensor newInput = input.slice(range);
                
                Tensor newK = newInput.matmul(keyWeightsTensor);
                Tensor newV = newInput.matmul(valueWeightsTensor);
                
                List<Tensor> newKTokens = TensorFactory.toList(newK);
                List<Tensor> newVTokens = TensorFactory.toList(newV);
                
                keyCache.addAll(newKTokens);
                valueCache.addAll(newVTokens);
            }
            
            K = TensorFactory.zeros(seqLength, headDimension);
            V = TensorFactory.zeros(seqLength, headDimension);
            
            int elementsToProcess = Math.min(seqLength, keyCache.size());
            
            for (int i = 0; i < elementsToProcess; i++) {
                Tensor kToken = keyCache.get(i);
                Tensor vToken = valueCache.get(i);
                
                for (int j = 0; j < headDimension; j++) {
                    K.set(kToken.get(0, j), i, j);
                    V.set(vToken.get(0, j), i, j);
                }
            }
        } else {
            K = input.matmul(keyWeightsTensor);
            V = input.matmul(valueWeightsTensor);
            
            clearCache();
            List<Tensor> kTokens = TensorFactory.toList(K);
            List<Tensor> vTokens = TensorFactory.toList(V);
            
            keyCache.addAll(kTokens);
            valueCache.addAll(vTokens);
        }

        double normalizer = Math.sqrt(headDimension);

        Tensor scores = Q.matmul(K.transpose()).div(normalizer);
        Tensor mask = TensorFactory.triangularMask(scores.shape()[0]);

        Tensor maskedScores = scores.add(mask);
        Tensor attentionWeights = maskedScores.softmax();

        return attentionWeights.matmul(V);
    }
}
