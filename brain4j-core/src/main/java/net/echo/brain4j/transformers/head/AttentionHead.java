package net.echo.brain4j.transformers.head;

import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.brain4j.structure.StatesCache;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;
import net.echo.math4j.math.tensor.index.Range;

import java.util.List;
import java.util.Random;

public class AttentionHead {

    protected int inputDimension;
    protected int headDimension;

    protected Tensor queryWeightsTensor;
    protected Tensor keyWeightsTensor;
    protected Tensor valueWeightsTensor;
    
    protected boolean useCache = true;

    public AttentionHead(int inputDimension, int headDimension) {
        this.inputDimension = inputDimension;
        this.headDimension = headDimension;

        this.queryWeightsTensor = TensorFactory.matrix(inputDimension, headDimension);
        this.keyWeightsTensor = TensorFactory.matrix(inputDimension, headDimension);
        this.valueWeightsTensor = TensorFactory.matrix(inputDimension, headDimension);
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
        Tensor Q = input.matmul(queryWeightsTensor);
        Tensor K = input.matmul(keyWeightsTensor);
        Tensor V = input.matmul(valueWeightsTensor);

        double normalizer = Math.sqrt(headDimension);

        Tensor scores = Q.matmul(K.transpose()).div(normalizer);
        Tensor attentionWeights = scores.softmax();

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
                
                List<Tensor> newKTokens = TensorFactory.toList(newK);
                List<Tensor> newVTokens = TensorFactory.toList(newV);
                
                localKeyCache.addAll(newKTokens);
                localValueCache.addAll(newVTokens);
            }
            
            K = TensorFactory.zeros(seqLength, headDimension);
            V = TensorFactory.zeros(seqLength, headDimension);
            
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
            
            List<Tensor> kTokens = TensorFactory.toList(K);
            List<Tensor> vTokens = TensorFactory.toList(V);
            
            localKeyCache.addAll(kTokens);
            localValueCache.addAll(vTokens);
        }
        
        double normalizer = Math.sqrt(headDimension);
        
        Tensor scores = Q.matmul(K.transpose()).div(normalizer);
        Tensor attentionWeights = scores.softmax();
        
        return attentionWeights.matmul(V);
    }
}
