package org.brain4j.core.transformers.head;

import org.brain4j.core.initialization.WeightInitializer;
import org.brain4j.core.structure.StatesCache;
import org.brain4j.math.tensor.Tensor;
import org.brain4j.math.tensor.Tensors;
import org.brain4j.math.tensor.index.Range;

import java.util.List;
import java.util.Random;

public class AttentionHead {

    protected int inputDimension;
    protected int headDimension;

    protected Tensor queryWeightsTensor;
    protected Tensor keyWeightsTensor;
    protected Tensor valueWeightsTensor;
    
    protected boolean useCache = true;

    public AttentionHead(int embeddingDim, int headDimension) {
        this.inputDimension = embeddingDim;
        this.headDimension = headDimension;

        this.queryWeightsTensor = Tensors.matrix(embeddingDim, headDimension);
        this.keyWeightsTensor = Tensors.matrix(embeddingDim, headDimension);
        this.valueWeightsTensor = Tensors.matrix(embeddingDim, headDimension);
    }

    public void compile(Random random, WeightInitializer initializer) {
        double bound = initializer.getBound(inputDimension, headDimension);

        for (int i = 0; i < inputDimension; i++) {
            for (int j = 0; j < headDimension; j++) {
                queryWeightsTensor.set(random.nextDouble(2 * bound) - bound, i, j);
                keyWeightsTensor.set(random.nextDouble(2 * bound) - bound, i, j);
                valueWeightsTensor.set(random.nextDouble(2 * bound) - bound, i, j);
            }
        }
    }

    public Tensor getQueryWeightsTensor() {
        return queryWeightsTensor;
    }

    public void setQueryWeightsTensor(Tensor queryWeightsTensor) {
        this.queryWeightsTensor = queryWeightsTensor;
    }

    public Tensor getKeyWeightsTensor() {
        return keyWeightsTensor;
    }

    public void setKeyWeightsTensor(Tensor keyWeightsTensor) {
        this.keyWeightsTensor = keyWeightsTensor;
    }

    public Tensor getValueWeightsTensor() {
        return valueWeightsTensor;
    }

    public void setValueWeightsTensor(Tensor valueWeightsTensor) {
        this.valueWeightsTensor = valueWeightsTensor;
    }

    public int size() {
        return 3 * inputDimension * headDimension;
    }

    public void setUseCache(boolean useCache) {
        this.useCache = useCache;
    }

    public boolean isUsingCache() {
        return useCache;
    }

    public Tensor attend(Tensor input) {
        // input = [seq_length, embedding_dim]
        Tensor Q = input.matmul(queryWeightsTensor); // [seq_length, head_dimension]
        Tensor K = input.matmul(keyWeightsTensor); // [seq_length, head_dimension]
        Tensor V = input.matmul(valueWeightsTensor); // [seq_length, head_dimension]

        double normalizer = Math.sqrt(headDimension);

        // [seq_length, seq_length]
        Tensor scores = Q.matmul(K.transpose()).div(normalizer);
        Tensor attentionWeights = scores.softmax();

        // [seq_length, head_dimension]
        return attentionWeights.matmul(V);
    }
    
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
        Tensor attentionWeights = scores.softmax();
        
        return attentionWeights.matmul(V);
    }
}
