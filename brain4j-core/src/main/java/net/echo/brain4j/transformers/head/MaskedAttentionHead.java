package net.echo.brain4j.transformers.head;

import net.echo.brain4j.structure.StatesCache;
import net.echo.math.tensor.Tensor;
import net.echo.math.tensor.Tensors;
import net.echo.math.tensor.index.Range;

import java.util.List;

public class MaskedAttentionHead extends AttentionHead {

    public MaskedAttentionHead(int inputDimension, int headDimension) {
        super(inputDimension, headDimension);
    }

    @Override
    public Tensor attend(Tensor input) {
        Tensor Q = input.matmul(queryWeightsTensor);
        Tensor K = input.matmul(keyWeightsTensor);
        Tensor V = input.matmul(valueWeightsTensor);

        double normalizer = Math.sqrt(headDimension);

        Tensor scores = Q.matmul(K.transpose()).div(normalizer);
        Tensor mask = Tensors.triangularMask(scores.shape()[0]);

        Tensor maskedScores = scores.add(mask);
        Tensor attentionWeights = maskedScores.softmax();

        return attentionWeights.matmul(V);
    }
    
    @Override
    public Tensor attend(StatesCache cache, Tensor input) {
        Tensor Q = input.matmul(queryWeightsTensor);
        
        int seqLength = input.shape()[0];

        List<Tensor> localKeyCache = cache.getKeyCacheForHead(this);
        List<Tensor> localValueCache = cache.getValueCacheForHead(this);
        
        int cachedLength = localKeyCache.size();
        
        Tensor K, V;

        if (isUsingCache() && cachedLength > 0) {
            if (seqLength > cachedLength) {
                Range range = new Range(cachedLength, seqLength);
                Tensor newInput = input.slice(range);
                
                Tensor newK = newInput.matmul(keyWeightsTensor);
                Tensor newV = newInput.matmul(valueWeightsTensor);
                
                List<Tensor> newKTokens = Tensors.toList(newK);
                List<Tensor> newVTokens = Tensors.toList(newV);
                
                localKeyCache.addAll(newKTokens);
                localValueCache.addAll(newVTokens);
            }
            
            K = Tensors.zeros(seqLength, headDimension);
            V = Tensors.zeros(seqLength, headDimension);
            
            int elementsToProcess = Math.min(seqLength, localKeyCache.size());
            
            for (int i = 0; i < elementsToProcess; i++) {
                Tensor kToken = localKeyCache.get(i);
                Tensor vToken = localValueCache.get(i);
                
                for (int j = 0; j < headDimension; j++) {
                    K.set(kToken.get(0, j), i, j);
                    V.set(vToken.get(0, j), i, j);
                }
            }
        } else {
            K = input.matmul(keyWeightsTensor);
            V = input.matmul(valueWeightsTensor);
            
            localKeyCache.clear();
            localValueCache.clear();
            
            List<Tensor> kTokens = Tensors.toList(K);
            List<Tensor> vTokens = Tensors.toList(V);
            
            localKeyCache.addAll(kTokens);
            localValueCache.addAll(vTokens);
        }
        
        double normalizer = Math.sqrt(headDimension);
        
        Tensor scores = Q.matmul(K.transpose()).div(normalizer);
        Tensor mask = Tensors.triangularMask(scores.shape()[0]);
        
        Tensor maskedScores = scores.add(mask);
        Tensor attentionWeights = maskedScores.softmax();
        
        return attentionWeights.matmul(V);
    }
}
