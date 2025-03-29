package net.echo.brain4j.transformers.head;

import net.echo.brain4j.model.initialization.WeightInitializer;
import net.echo.math4j.math.tensor.Tensor;
import net.echo.math4j.math.tensor.TensorFactory;
import net.echo.math4j.math.tensor.index.Range;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class AttentionHead {

    protected final int inputDimension;
    protected final int headDimension;

    protected final Tensor queryWeightsTensor;
    protected final Tensor keyWeightsTensor;
    protected final Tensor valueWeightsTensor;
    
    protected List<Tensor> keyCache;
    protected List<Tensor> valueCache;
    protected boolean useCache;

    public AttentionHead(int inputDimension, int headDimension) {
        this.inputDimension = inputDimension;
        this.headDimension = headDimension;

        this.queryWeightsTensor = TensorFactory.matrix(inputDimension, headDimension);
        this.keyWeightsTensor = TensorFactory.matrix(inputDimension, headDimension);
        this.valueWeightsTensor = TensorFactory.matrix(inputDimension, headDimension);
        
        this.keyCache = new ArrayList<>();
        this.valueCache = new ArrayList<>();
        this.useCache = true;
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

    public int size() {
        return 3 * inputDimension * headDimension;
    }

    public void setUseCache(boolean useCache) {
        this.useCache = useCache;
    }

    public boolean isUsingCache() {
        return useCache;
    }

    public void clearCache() {
        keyCache.clear();
        valueCache.clear();
    }

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
        Tensor attentionWeights = scores.softmax();

        return attentionWeights.matmul(V);
    }
}
